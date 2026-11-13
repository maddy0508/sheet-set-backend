from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import math
import shapely.geometry as sg
import shapely.ops
from shapely.geometry.polygon import orient
from compas.geometry import Polygon
from compas_cgal.straight_skeleton_2 import extrude

app = FastAPI(title="Sheet Set API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

EPS = 1e-6

# ==========================================
# DATA MODELS
# ==========================================
class RoofRequest(BaseModel):
    exterior: List[List[float]] = Field(..., description="List of [x, y] points forming the outer boundary")
    interiors: Optional[List[List[List[float]]]] = Field(None, description="Optional list of interior holes")
    pitch_deg: float = Field(30.0, description="Roof pitch in degrees")
    sheet_width_mm: float = Field(760.0, description="Metal sheet width in mm")
    max_stock_mm: float = Field(12000.0, description="Maximum coil length in mm")
    overhang_mm: float = Field(150.0, description="Eave overhang in mm")

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

# ==========================================
# ROOF ENGINE CLASS
# ==========================================
class RoofEngine:
    def __init__(self, exterior_2d, interiors_2d=None, pitch_deg=30.0, 
                 sheet_width_mm=760.0, max_stock_mm=12000.0, overhang_mm=150.0):
        # Enforce CCW winding for exterior before doing anything else
        self.ext_2d = self._ensure_ccw(self._clean_ring(exterior_2d))
        if len(self.ext_2d) < 3:
            raise ValueError("Exterior must have at least 3 distinct points.")
            
        self.int_2d_list = []
        if interiors_2d:
            for interior in interiors_2d:
                # Enforce CW winding for interiors
                cleaned = self._ensure_cw(self._clean_ring(interior))
                if len(cleaned) >= 3:
                    self.int_2d_list.append(cleaned)
        
        self.pitch_rad = math.radians(pitch_deg)
        self.sheet_width_m = sheet_width_mm / 1000.0
        self.max_stock_m = max_stock_mm / 1000.0
        self.overhang_m = overhang_mm / 1000.0
        
        self.faces_2d = []
        self.sheets = []
        self.warnings = []

    def _clean_ring(self, coords):
        cleaned = []
        for p in coords:
            pt = (float(p[0]), float(p[1]))
            if not cleaned or math.dist(cleaned[-1], pt) > EPS:
                cleaned.append(pt)
        if len(cleaned) > 1 and math.dist(cleaned[0], cleaned[-1]) < EPS:
            cleaned.pop()
        return cleaned

    def _ensure_ccw(self, ring):
        """Ensure ring is counter-clockwise. Returns reversed ring if needed."""
        area2 = sum(
            ring[i][0] * ring[(i+1) % len(ring)][1] - ring[(i+1) % len(ring)][0] * ring[i][1]
            for i in range(len(ring))
        )
        return ring if area2 > 0 else ring[::-1]

    def _ensure_cw(self, ring):
        """Ensure ring is clockwise. Returns reversed ring if needed."""
        area2 = sum(
            ring[i][0] * ring[(i+1) % len(ring)][1] - ring[(i+1) % len(ring)][0] * ring[i][1]
            for i in range(len(ring))
        )
        return ring if area2 < 0 else ring[::-1]

    def run_engine_1_geometry(self):
        ext_3d = [(x, y, 0.0) for x, y in self.ext_2d]
        ints_3d = [[(x, y, 0.0) for x, y in ring] for ring in self.int_2d_list]
        
        compas_poly = Polygon(ext_3d)
        angles = math.degrees(self.pitch_rad)
        holes = ints_3d if ints_3d else None
        
        try:
            mesh, lines_3d = extrude(compas_poly, holes=holes, angles=angles)
        except Exception as e:
            raise ValueError(f"CGAL Skeleton generation failed: {e}")

        skeleton_lines = [
            sg.LineString([(l.start[0], l.start[1]), (l.end[0], l.end[1])]) 
            for l in lines_3d
        ]
        
        shapely_poly = sg.Polygon(self.ext_2d, self.int_2d_list)
        shapely_poly = orient(shapely_poly, sign=1.0)
        
        return shapely_poly, skeleton_lines

    def _get_cut_angle(self, pt, polygon_coords):
        for i in range(len(polygon_coords) - 1):
            p1, p2 = polygon_coords[i], polygon_coords[i+1]
            if sg.LineString([p1, p2]).distance(sg.Point(pt)) < EPS:
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                if dx == 0 and dy == 0:
                    continue
                angle_rad = math.atan2(dx, dy)
                return round(90.0 - abs(math.degrees(angle_rad)), 1)
        self.warnings.append(f"Cut angle lookup defaulted to 0° at point {pt}")
        return 0.0

    def _inverse_transform(self, pt, p1, angle, stretch):
        u, v = pt
        v_orig = v / stretch
        return (
            p1[0] + u * math.cos(angle) - v_orig * math.sin(angle),
            p1[1] + u * math.sin(angle) + v_orig * math.cos(angle)
        )

    def run_engine_2_laydown(self, footprint_poly, skeleton_lines):
        boundary_lines = []
        for ring in [footprint_poly.exterior] + list(footprint_poly.interiors):
            coords = list(ring.coords)
            for i in range(len(coords) - 1):
                boundary_lines.append(sg.LineString([coords[i], coords[i+1]]))

        self.faces_2d = list(shapely.ops.polygonize(
            shapely.ops.unary_union(skeleton_lines + boundary_lines)
        ))
        if not self.faces_2d:
            raise ValueError("Polygonization failed. Check footprint is valid and closed.")

        for face in self.faces_2d:
            coords = list(face.exterior.coords)[:-1]
            eave_idx = -1
            max_eave = 0
            
            for i in range(len(coords)):
                p1, p2 = coords[i], coords[(i+1) % len(coords)]
                edge = sg.LineString([p1, p2])
                if footprint_poly.boundary.distance(edge) < EPS and edge.length > max_eave:
                    max_eave = edge.length
                    eave_idx = i
            
            if eave_idx == -1:
                self.warnings.append("Internal face with no eave — using longest edge as reference")
                max_len = 0
                for i in range(len(coords)):
                    p1, p2 = coords[i], coords[(i+1) % len(coords)]
                    ln = math.dist(p1, p2)
                    if ln > max_len:
                        max_len = ln
                        eave_idx = i

            p1 = coords[eave_idx]
            p2 = coords[(eave_idx + 1) % len(coords)]
            angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
            
            rotated_coords = []
            for x, y in coords:
                nx = (x - p1[0]) * math.cos(-angle) - (y - p1[1]) * math.sin(-angle)
                ny = (x - p1[0]) * math.sin(-angle) + (y - p1[1]) * math.cos(-angle)
                rotated_coords.append((nx, ny))
                
            stretch = 1.0 / math.cos(self.pitch_rad)
            stretched_coords = [(x, y * stretch) for x, y in rotated_coords]
            face_stretched = sg.Polygon(stretched_coords)
            stretched_poly_coords = list(face_stretched.exterior.coords)
            
            minx, miny, maxx, maxy = face_stretched.bounds
            x = minx
            
            while x <= maxx:
                line = sg.LineString([(x, miny - 1), (x, maxy + 1)])
                clipped = face_stretched.intersection(line)
                
                if not clipped.is_empty and clipped.geom_type == 'LineString':
                    if clipped.length > 0.3: 
                        pt_start = clipped.coords[0]
                        pt_end = clipped.coords[-1]
                        
                        start_orig = self._inverse_transform(pt_start, p1, angle, stretch)
                        end_orig = self._inverse_transform(pt_end, p1, angle, stretch)
                        
                        overhang_start = (
                            self.overhang_m 
                            if footprint_poly.boundary.distance(sg.Point(start_orig)) < EPS 
                            else 0.0
                        )
                        overhang_end = (
                            self.overhang_m 
                            if footprint_poly.boundary.distance(sg.Point(end_orig)) < EPS 
                            else 0.0
                        )
                        
                        true_length = clipped.length + overhang_start + overhang_end
                        
                        angle_start = self._get_cut_angle(pt_start, stretched_poly_coords)
                        angle_end = self._get_cut_angle(pt_end, stretched_poly_coords)
                        
                        self.sheets.append({
                            'start': list(start_orig),
                            'end': list(end_orig),
                            'length_m': true_length,
                            'length_mm': int(round(true_length * 1000)),
                            'angle_start': angle_start,
                            'angle_end': angle_end
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
            if not placed:
                stock.append({
                    'remaining': self.max_stock_m - sheet['length_m'],
                    'cuts': [sheet]
                })
        return stock

    def get_summary_and_csv(self, stock):
        total_sheets = len(self.sheets)
        total_used = sum(s['length_m'] for s in self.sheets)
        total_stock = len(stock) * self.max_stock_m
        waste_pct = ((total_stock - total_used) / total_stock) * 100 if total_stock > 0 else 0
        
        summary = "--- ORDER SUMMARY ---\n"
        summary += f"Total Sheets: {total_sheets}\n"
        summary += f"Total 12m Coils: {len(stock)}\n"
        summary += f"Linear Meters Used: {total_used:.2f}m\n"
        summary += f"Linear Meters Ordered: {total_stock:.2f}m\n"
        summary += f"Waste: {waste_pct:.1f}%\n"
        if self.warnings:
            summary += f"Warnings: {len(self.warnings)}\n"
            for w in self.warnings:
                summary += f"  ⚠️ {w}\n"
                
        csv_lines = ["SheetID,Length_mm,CutAngle_Bottom,CutAngle_Top,Stock_Coil"]
        for i, s in enumerate(stock):
            for j, cut in enumerate(s['cuts']):
                csv_lines.append(
                    f"P{i:02d}{j:02d},{cut['length_mm']},{cut['angle_start']},{cut['angle_end']},Coil_{i+1}"
                )
        
        return summary, "\n".join(csv_lines)

# ==========================================
# API ENDPOINT
# ==========================================
@app.post("/calculate", response_model=RoofResponse)
def calculate_roof(req: RoofRequest):
    try:
        engine = RoofEngine(
            exterior_2d=req.exterior,
            interiors_2d=req.interiors,
            pitch_deg=req.pitch_deg,
            sheet_width_mm=req.sheet_width_mm,
            max_stock_mm=req.max_stock_mm,
            overhang_mm=req.overhang_mm
        )
        poly, skel = engine.run_engine_1_geometry()
        engine.run_engine_2_laydown(poly, skel)
        stock = engine.run_engine_3_nesting()
        summary, csv_data = engine.get_summary_and_csv(stock)
        
        return RoofResponse(
            summary=summary,
            csv_data=csv_data,
            sheets=[SheetData(**s) for s in engine.sheets],
            total_faces=len(engine.faces_2d)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Sheet Set API is running"}
