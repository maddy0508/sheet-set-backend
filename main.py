from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import math
import shapely.geometry as sg
import shapely.ops
from shapely.geometry.polygon import orient
from compas.geometry import Polygon
from compas_cgal.straight_skeleton_2 import extrude
import requests
import os
import io
import numpy as np
import cv2
from PIL import Image
from ultralytics import SAM
import torch

app = FastAPI(title="Sheet Set API", version="1.0.0")

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["*"], allow_headers=["*"])

EPS = 1e-6

# --- LOAD MOBILESAM MODEL ---
# This downloads mobile_sam.pt (~40MB) automatically on first run
print("Loading MobileSAM model...")
sam_model = SAM("mobile_sam.pt")

class RoofRequest(BaseModel):
    exterior: List[List[float]] = Field(...)
    interiors: Optional[List[List[List[float]]]] = Field(None)
    pitch_deg: float = Field(30.0)
    sheet_width_mm: float = Field(760.0)
    max_stock_mm: float = Field(12000.0)
    overhang_mm: float = Field(150.0)

class SheetData(BaseModel):
    start: List[float]
    end: List[float]
    length_m: float
    length_mm: int
    angle_start: float
    angle_end: float

class RoofResponse(BaseModel):
    summary: str
    csv_data: str
    sheets: List[SheetData]
    total_faces: int
    total_coils: int
    total_stock_m: float
    mesh_vertices: List[List[float]]
    mesh_faces: List[List[int]]
    roof_area_m2: float
    total_waste_pct: float
    trims: dict
    warnings: List[str]

# --- AI SEGMENTATION ENDPOINT ---
@app.get("/segment-roof")
async def segment_roof(lat: float = Query(...), lng: float = Query(...), zoom: int = 19):
    try:
        # 1. Convert Lat/Lng to Tile Coordinates
        n = 2.0 ** zoom
        xtile = int((lng + 180.0) / 360.0 * n)
        ytile = int((1.0 - math.log(math.tan(math.radians(lat)) + (1 / math.cos(math.radians(lat)))) / math.pi) / 2.0 * n)
        
        # 2. Fetch the Google Satellite Tile
        tile_url = f"https://mt1.google.com/vt/lyrs=s&x={xtile}&y={ytile}&z={zoom}"
        response = requests.get(tile_url, timeout=10)
        response.raise_for_status()
        
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        img_np = np.array(img)
        
        # 3. Calculate the pixel coordinate of the tap within the 256x256 tile
        # Formula: (lng - tile_lng_start) / tile_lng_width * 256
        tile_lng_start = xtile / n * 360.0 - 180.0
        tile_lat_start = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * ytile / n))))
        tile_lng_end = (xtile + 1) / n * 360.0 - 180.0
        tile_lat_end = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (ytile + 1) / n))))
        
        px = int((lng - tile_lng_start) / (tile_lng_end - tile_lng_start) * 256)
        py = int((tile_lat_start - lat) / (tile_lat_start - tile_lat_end) * 256)
        
        px = max(0, min(255, px))
        py = max(0, min(255, py))

        # 4. Run MobileSAM
        results = sam_model(img_np, points=[[px, py]], labels=[1])
        
        if not results or len(results[0].masks) == 0:
            raise HTTPException(status_code=404, detail="Could not detect a roof from this point. Try tapping closer to the center.")
            
        # 5. Extract Mask and Convert to Polygon
        mask = results[0].masks.data[0].cpu().numpy()
        mask = (mask > 0.5).astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise HTTPException(status_code=404, detail="Mask generated but no polygon found.")
            
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Simplify polygon slightly (epsilon = 0.02 * arcLength)
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # 6. Convert Pixel Coordinates back to Lat/Lng
        polygon_latlng = []
        for point in approx:
            x, y = point[0]
            lng_px = tile_lng_start + (x / 256.0) * (tile_lng_end - tile_lng_start)
            lat_py = tile_lat_start - (y / 256.0) * (tile_lat_start - tile_lat_end)
            polygon_latlng.append({"lat": lat_py, "lng": lng_px})
            
        return {"polygon": polygon_latlng}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- REST OF YOUR ROOF ENGINE (Unchanged) ---
OVERPASS_SERVERS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter"
]

@app.get("/get-footprint")
async def get_footprint(lat: float = Query(...), lng: float = Query(...)):
    query = f"[out:json];way(around:30,{lat},{lng})['building'];out geom;"
    last_error = "No building found at this location in OpenStreetMap."
    for server in OVERPASS_SERVERS:
        try:
            res = requests.post(server, data={"data": query}, timeout=8)
            res.raise_for_status()
            data = res.json()
            if not data.get('elements'): continue
            closest_poly = None
            min_dist = float('inf')
            for element in data['elements']:
                coords = [(node['lat'], node['lon']) for node in element.get('geometry', [])]
                if len(coords) >= 3:
                    poly = sg.Polygon(coords)
                    pt = sg.Point(lat, lng)
                    dist = poly.distance(pt)
                    if dist < min_dist:
                        min_dist = dist
                        closest_poly = coords
            if closest_poly:
                return {"polygon": [{"lat": c[0], "lng": c[1]} for c in closest_poly]}
        except requests.exceptions.RequestException:
            last_error = f"Server busy or offline. Try drawing manually."
            continue
    return {"error": last_error}

class RoofEngine:
    def __init__(self, exterior_2d, interiors_2d=None, pitch_deg=30.0, sheet_width_mm=760.0, max_stock_mm=12000.0, overhang_mm=150.0):
        self.warnings = []
        self.pitch_deg = max(5.0, min(75.0, pitch_deg))
        if self.pitch_deg != pitch_deg:
            self.warnings.append(f"Pitch clamped to {self.pitch_deg}° for safety.")
        self.ext_2d = self._ensure_ccw(self._clean_ring(exterior_2d))
        poly = sg.Polygon(self.ext_2d)
        if not poly.is_valid:
            poly = poly.buffer(0)
            self.warnings.append("Shape was self-intersecting and has been auto-healed.")
            if poly.geom_type == 'Polygon':
                self.ext_2d = list(poly.exterior.coords)[:-1]
            elif poly.geom_type == 'MultiPolygon':
                largest = max(poly.geoms, key=lambda p: p.area)
                self.ext_2d = list(largest.exterior.coords)[:-1]
        self.int_2d_list = []
        if interiors_2d:
            for interior in interiors_2d:
                cleaned = self._ensure_cw(self._clean_ring(interior))
                if len(cleaned) >= 3: self.int_2d_list.append(cleaned)
                else: self.warnings.append("An interior hole was too small and was ignored.")
        self.pitch_rad = math.radians(self.pitch_deg)
        self.sheet_width_m = sheet_width_mm / 1000.0
        self.max_stock_m = max_stock_mm / 1000.0
        self.overhang_m = overhang_mm / 1000.0
        self.faces_2d, self.sheets = [], []

    def _clean_ring(self, coords):
        cleaned = []
        for p in coords:
            pt = (float(p[0]), float(p[1]))
            if not cleaned or math.dist(cleaned[-1], pt) > EPS: cleaned.append(pt)
        if len(cleaned) > 1 and math.dist(cleaned[0], cleaned[-1]) < EPS: cleaned.pop()
        return cleaned

    def _ensure_ccw(self, ring):
        area2 = sum(ring[i][0] * ring[(i+1) % len(ring)][1] - ring[(i+1) % len(ring)][0] * ring[i][1] for i in range(len(ring)))
        return ring if area2 > 0 else ring[::-1]

    def _ensure_cw(self, ring):
        area2 = sum(ring[i][0] * ring[(i+1) % len(ring)][1] - ring[(i+1) % len(ring)][0] * ring[i][1] for i in range(len(ring)))
        return ring if area2 < 0 else ring[::-1]

    def run_engine_1_geometry(self):
        ext_3d = [(x, y, 0.0) for x, y in self.ext_2d]
        ints_3d = [[(x, y, 0.0) for x, y in ring] for ring in self.int_2d_list]
        compas_poly = Polygon(ext_3d)
        angles = math.degrees(self.pitch_rad)
        holes = ints_3d if ints_3d else None
        try: mesh, lines_3d = extrude(compas_poly, holes=holes, angles=angles)
        except Exception as e: raise ValueError(f"CGAL_SKELETON_FAILED")
        verts_raw, faces_raw = mesh.to_vertices_and_faces()
        mesh_verts = [list(v) for v in verts_raw]
        mesh_faces = [list(f) for f in faces_raw]
        skeleton_lines = [sg.LineString([(l.start[0], l.start[1]), (l.end[0], l.end[1])]) for l in lines_3d]
        shapely_poly = orient(sg.Polygon(self.ext_2d, self.int_2d_list), sign=1.0)
        self.skeleton_lines = lines_3d
        return shapely_poly, skeleton_lines, mesh_verts, mesh_faces

    def _get_cut_angle(self, pt, polygon_coords):
        for i in range(len(polygon_coords) - 1):
            p1, p2 = polygon_coords[i], polygon_coords[i+1]
            if sg.LineString([p1, p2]).distance(sg.Point(pt)) < EPS:
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                if dx == 0 and dy == 0: continue
                return round(90.0 - abs(math.degrees(math.atan2(dx, dy))), 1)
        return 0.0

    def _inverse_transform(self, pt, p1, angle, stretch):
        u, v = pt
        v_orig = v / stretch
        return (p1[0] + u * math.cos(angle) - v_orig * math.sin(angle), p1[1] + u * math.sin(angle) + v_orig * math.cos(angle))

    def run_engine_2_laydown(self, footprint_poly, skeleton_lines):
        boundary_lines = []
        for ring in [footprint_poly.exterior] + list(footprint_poly.interiors):
            coords = list(ring.coords)
            for i in range(len(coords) - 1):
                boundary_lines.append(sg.LineString([coords[i], coords[i+1]]))
        self.faces_2d = list(shapely.ops.polygonize(shapely.ops.unary_union(skeleton_lines + boundary_lines)))
        if not self.faces_2d: raise ValueError("POLYGONIZATION_FAILED")
        for face in self.faces_2d:
            coords = list(face.exterior.coords)[:-1]
            eave_idx, max_eave = -1, 0
            for i in range(len(coords)):
                p1, p2 = coords[i], coords[(i+1) % len(coords)]
                edge = sg.LineString([p1, p2])
                if footprint_poly.boundary.distance(edge) < EPS and edge.length > max_eave:
                    max_eave, eave_idx = edge.length, i
            if eave_idx == -1:
                max_len = 0
                for i in range(len(coords)):
                    p1, p2 = coords[i], coords[(i+1) % len(coords)]
                    if math.dist(p1, p2) > max_len: max_len, eave_idx = math.dist(p1, p2), i
            p1 = coords[eave_idx]
            p2 = coords[(eave_idx+1) % len(coords)]
            angle = math.atan2(p2[1]-p1[1], p2[0]-p1[0])
            rotated_coords = [((x - p1[0]) * math.cos(-angle) - (y - p1[1]) * math.sin(-angle), (x - p1[0]) * math.sin(-angle) + (y - p1[1]) * math.cos(-angle)) for x, y in coords]
            stretch = 1.0 / math.cos(self.pitch_rad)
            face_stretched = sg.Polygon([(x, y * stretch) for x, y in rotated_coords])
            stretched_poly_coords = list(face_stretched.exterior.coords)
            minx, miny, maxx, maxy = face_stretched.bounds
            x = minx
            while x <= maxx:
                clipped = face_stretched.intersection(sg.LineString([(x, miny - 1), (x, maxy + 1)]))
                if not clipped.is_empty and clipped.geom_type == 'LineString' and clipped.length > 0.3:
                    pt_start, pt_end = clipped.coords[0], clipped.coords[-1]
                    overhang_start = self.overhang_m if footprint_poly.boundary.distance(sg.Point(self._inverse_transform(pt_start, p1, angle, stretch))) < EPS else 0.0
                    overhang_end = self.overhang_m if footprint_poly.boundary.distance(sg.Point(self._inverse_transform(pt_end, p1, angle, stretch))) < EPS else 0.0
                    self.sheets.append({
                        'start': list(self._inverse_transform(pt_start, p1, angle, stretch)),
                        'end': list(self._inverse_transform(pt_end, p1, angle, stretch)),
                        'length_m': clipped.length + overhang_start + overhang_end,
                        'length_mm': int(round((clipped.length + overhang_start + overhang_end) * 1000)),
                        'angle_start': self._get_cut_angle(pt_start, stretched_poly_coords),
                        'angle_end': self._get_cut_angle(pt_end, stretched_poly_coords)
                    })
                x += self.sheet_width_m

    def run_engine_3_nesting(self):
        sorted_sheets = sorted(self.sheets, key=lambda s: s['length_m'], reverse=True)
        stock = []
        for sheet in sorted_sheets:
            placed = False
            for s in stock:
                if s['remaining'] >= sheet['length_m']:
                    s['remaining'] -= sheet['length_m']
                    s['cuts'].append(sheet)
                    placed = True
                    break
            if not placed: stock.append({'remaining': self.max_stock_m - sheet['length_m'], 'cuts': [sheet]})
        return stock

    def get_results(self, stock, footprint_poly):
        total_sheets = len(self.sheets)
        total_used = sum(s['length_m'] for s in self.sheets)
        total_stock = len(stock) * self.max_stock_m
        waste_pct = ((total_stock - total_used) / total_stock) * 100 if total_stock > 0 else 0
        plan_area = footprint_poly.area
        slope_factor = 1.0 / math.cos(self.pitch_rad)
        roof_area_m2 = plan_area * slope_factor
        ridge_len = 0.0; hip_len = 0.0; valley_len = 0.0; ridge_count = 0; hip_count = 0
        coords = list(footprint_poly.exterior.coords)[:-1]
        reflex_corners = []
        for i in range(len(coords)):
            p1, p2, p3 = coords[i-1], coords[i], coords[(i+1)%len(coords)]
            angle = math.atan2(p3[1]-p2[1], p3[0]-p2[0]) - math.atan2(p1[1]-p2[1], p1[0]-p2[0])
            while angle < 0: angle += 2*math.pi
            if angle > math.pi: reflex_corners.append(sg.Point(p2))
        for line in self.skeleton_lines:
            z1, z2 = line.start[2], line.end[2]
            length = math.dist((line.start[0], line.start[1]), (line.end[0], line.end[1]))
            if abs(z1 - z2) < EPS: ridge_len += length; ridge_count += 1
            else:
                start_pt = sg.Point(line.start[0], line.start[1])
                end_pt = sg.Point(line.end[0], line.end[1])
                is_valley = any(c.distance(start_pt) < 0.1 or c.distance(end_pt) < 0.1 for c in reflex_corners)
                if is_valley: valley_len += length
                else: hip_len += length; hip_count += 1
        eave_len = footprint_poly.length
        hip_len = hip_len * slope_factor; valley_len = valley_len * slope_factor
        ridge_len *= 1.05; hip_len *= 1.05; valley_len *= 1.05
        trims = {"hip_capping_m": round(hip_len, 1), "hip_count": hip_count, "ridge_capping_m": round(ridge_len, 1), "ridge_count": ridge_count, "valley_capping_m": round(valley_len, 1), "eaves_gutter_m": round(eave_len, 1)}
        summary = f"Total Sheets: {total_sheets}\nTotal Coils: {len(stock)}\nLinear Meters Used: {total_used:.2f}m\nRoof Area: {roof_area_m2:.1f} m²\nWaste: {waste_pct:.1f}%"
        csv_lines = ["SheetID,Length_mm,CutAngle_Bottom,CutAngle_Top,Stock_Coil"]
        for i, s in enumerate(stock):
            for j, cut in enumerate(s['cuts']):
                csv_lines.append(f"P{i:02d}{j:02d},{cut['length_mm']},{cut['angle_start']},{cut['angle_end']},Coil_{i+1}")
        return summary, "\n".join(csv_lines), roof_area_m2, waste_pct, trims, len(stock), total_stock

@app.post("/calculate", response_model=RoofResponse)
def calculate_roof(req: RoofRequest):
    try:
        engine = RoofEngine(req.exterior, req.interiors, req.pitch_deg, req.sheet_width_mm, req.max_stock_mm, req.overhang_mm)
        poly, skel, mesh_verts, mesh_faces = engine.run_engine_1_geometry()
        engine.run_engine_2_laydown(poly, skel)
        stock = engine.run_engine_3_nesting()
        summary, csv_data, area, waste, trims, coils, stock_m = engine.get_results(stock, poly)
        return RoofResponse(summary=summary, csv_data=csv_data, sheets=[SheetData(**s) for s in engine.sheets], total_faces=len(engine.faces_2d), mesh_vertices=mesh_verts, mesh_faces=mesh_faces, roof_area_m2=area, total_waste_pct=waste, trims=trims, total_coils=coils, total_stock_m=stock_m, warnings=engine.warnings)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health_check(): return {"status": "ok"}
