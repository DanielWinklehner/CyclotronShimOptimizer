"""Export pole geometry as Inventor VBA macro."""

import numpy as np
import os
from typing import Union


class InventorPoleExporter:
    """Generate Inventor VBA macro for cyclotron pole."""

    def __init__(self, config, rank: int = 0, verbosity: int = 0):
        """
        Initialize exporter.

        :param config: CyclotronConfig
        :param rank: MPI rank
        :param verbosity: Verbosity level
        """
        self.config = config
        self.rank = rank
        self.verbosity = verbosity

    def export_macro(self,
                     pole_shape,
                     output_path: str = 'output/cyclotron_pole.bas') -> Union[str, None]:
        """
        Generate Inventor VBA macro to reconstruct the pole.

        For each radial segment:
        1. Create 2D annulus polygon (from inner to outer radius)
        2. Extrude to maximum top offset height
        3. Apply chamfer to top edge for angled surface

        :param pole_shape: PoleShape object with shim parameters
        :param output_path: Output .bas file path
        :return: Path to exported macro file
        """

        if self.rank > 0:
            return None

        if self.verbosity >= 1:
            print(f"\nGenerating Inventor VBA macro...", flush=True)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        cfg = self.config
        pole_cfg = cfg.pole
        # side_cfg = cfg.side_shim
        # top_cfg = cfg.top_shim

        # Extract shim data
        side_offsets_deg = pole_shape.get_side_offsets_deg()
        top_offsets_mm = pole_shape.get_top_offsets_mm()
        n_segments = pole_shape.num_segments

        # Generate macro
        macro_code = self._generate_macro_code(
            side_offsets_deg, top_offsets_mm, n_segments,
            pole_cfg
        )

        # Write to file
        with open(output_path, 'w') as f:
            f.write(macro_code)

        if self.verbosity >= 1:
            print(f"✓ Inventor macro generated: {output_path}", flush=True)
            print(f"  ├─ Radial segments: {n_segments}", flush=True)
            print(f"  ├─ Inner radius: {pole_cfg.inner_radius_mm} mm", flush=True)
            print(f"  ├─ Outer radius: {pole_cfg.outer_radius_mm} mm", flush=True)
            print(f"  ├─ Pole height: {pole_cfg.height_mm} mm", flush=True)
            print(f"  └─ Copy into Inventor: Tools → Macros → Edit/Run\n", flush=True)

        return output_path

    @staticmethod
    def _generate_array_assignments(array_name: str, values: np.ndarray) -> str:
        """
        Generate VBA code to populate array element-by-element.

        :param array_name: Name of the array variable
        :param values: Array of values
        :return: VBA assignment code
        """
        lines = []
        for i, val in enumerate(values):
            val *= 0.1  # Radia uses mm, VB Macro expects cm
            lines.append(f"    {array_name}({i}) = {val:.6f}")
        return "\n".join(lines)

    def _generate_macro_code(self, side_offsets_deg, top_offsets_mm, n_segments,
                             pole_cfg) -> str:
        """Generate complete VBA macro code for Inventor."""

        # Pole parameters
        inner_rad = pole_cfg.inner_radius_mm
        outer_rad = pole_cfg.outer_radius_mm
        pole_height = pole_cfg.height_mm
        pole_angle_deg = pole_cfg.full_angle_deg
        pole_half_angle_deg = pole_angle_deg / 2.0

        # Generate VBA array assignment code
        side_assignments = self._generate_array_assignments("sideShims", side_offsets_deg)
        top_assignments = self._generate_array_assignments("topShims", top_offsets_mm)

        macro = f'''
    ' ============================================================================
    ' Cyclotron Pole Geometry - Auto-generated Inventor Macro
    ' Generated from optimization results
    ' ============================================================================
    '
    ' Procedure:
    ' For each radial segment i (from 0 to n_segments-1):
    '   1. Create 2D sketch of annulus (inner to outer radius)
    '   2. Extrude to max(top_offset[i], top_offset[i+1])
    '   3. Apply chamfer to top edge for angled surface
    '
    ' ============================================================================

    Option Explicit

    Function IsClose(val1 As Double, val2 As Double) As Boolean
        IsClose = (Abs(val1 - val2) < 0.001)  ' Within 0.1 mm
    End Function


    Sub DrawAnnulusSegment(oSketch As PlanarSketch, _
                           r_inner As Double, r_outer As Double, _
                           ang_start As Double, ang_end As Double)

        Dim oSketchLines As SketchLines
        Dim oArcs As SketchArcs
        Dim pt1 As Point2d, pt2 As Point2d, pt3 As Point2d, pt4 As Point2d
        Dim ptCenter As Point2d

        Dim line1 As SketchLine
        Dim line2 As SketchLine
        Dim arc1 As SketchArc
        Dim arc2 As SketchArc

        Set oSketchLines = oSketch.SketchLines
        Set oArcs = oSketch.SketchArcs
        Set ptCenter = ThisApplication.TransientGeometry.CreatePoint2d(0, 0)

        ' Vertices of annulus segment
        ' pt1: inner radius at 0
        Set pt1 = ThisApplication.TransientGeometry.CreatePoint2d( _
            r_inner, 0#)

        ' pt2: outer radius at 0
        Set pt2 = ThisApplication.TransientGeometry.CreatePoint2d( _
            r_outer, 0#)

        ' pt3: outer radius at end angle
        Set pt3 = ThisApplication.TransientGeometry.CreatePoint2d( _
            r_outer * Cos(ang_end), r_outer * Sin(ang_end))

        ' pt4: inner radius at start angle
        Set pt4 = ThisApplication.TransientGeometry.CreatePoint2d( _
            r_inner * Cos(ang_start), r_inner * Sin(ang_start))

        ' Draw outer arc (from pt2 to pt3, centered at origin)
        Set arc1 = oArcs.AddByCenterStartEndPoint(ptCenter, pt2, pt3)

        ' Draw line from outer to inner at end angle
        Set line1 = oSketchLines.AddByTwoPoints(pt3, pt4)

        ' Draw inner arc (from pt4 to pt1, centered at origin, reversed)
        Set arc2 = oArcs.AddByCenterStartEndPoint(ptCenter, pt1, pt4)

        ' Close the loop: line from inner to outer at start angle
        Set line2 = oSketchLines.AddByTwoPoints(pt1, pt2)

        Call arc1.EndSketchPoint.Merge(line1.StartSketchPoint)
        Call arc2.EndSketchPoint.Merge(line1.EndSketchPoint)
        Call line2.EndSketchPoint.Merge(arc1.StartSketchPoint)
        Call line2.StartSketchPoint.Merge(arc2.StartSketchPoint)

    End Sub

    Sub ApplyTopChamfer(oCompDef As PartComponentDefinition, _
               oExtrude As ExtrudeFeature, _
               chamferRadius1 As Double, _
               chamferRadius2 As Double, _
               chamferInner As Boolean)

        ' Apply a two-distance chamfer to the top edge
        ' chamferRadius1: radial distance (between segment edges)
        ' chamferRadius2: vertical distance (height difference between segments)

        Dim oChamferFeature As ChamferFeature
        Dim oChamferDef As ChamferDefinition
        Dim i As Integer
        Dim j As Integer

        ' Find and chamfer the top edges
        Dim oTopFace As Face
        Set oTopFace = oExtrude.EndFaces.Item(1)
        Dim oChamferEdgeOptions As EdgeCollection
        Set oChamferEdgeOptions = ThisApplication.TransientObjects.CreateEdgeCollection
        Dim oChamferEdge As EdgeCollection
        Set oChamferEdge = ThisApplication.TransientObjects.CreateEdgeCollection

        ' Boil it down to inner and outer arc of top surface
        For i = 1 To oTopFace.Edges.Count
            If oTopFace.Edges(i).GeometryType = kCircularArcCurve Then
                Call oChamferEdgeOptions.Add(oTopFace.Edges.Item(i))
            End If
        Next i

        If chamferInner Then
            If oChamferEdgeOptions(1).Geometry.Radius < oChamferEdgeOptions(2).Geometry.Radius Then
                Call oChamferEdge.Add(oChamferEdgeOptions.Item(1))
            Else
                Call oChamferEdge.Add(oChamferEdgeOptions.Item(2))
            End If
        Else
            If oChamferEdgeOptions(1).Geometry.Radius > oChamferEdgeOptions(2).Geometry.Radius Then
                Call oChamferEdge.Add(oChamferEdgeOptions.Item(1))
            Else
                Call oChamferEdge.Add(oChamferEdgeOptions.Item(2))
            End If
        End If

        Call oCompDef.Features.ChamferFeatures.AddUsingTwoDistances(oChamferEdge, oTopFace, chamferRadius1, chamferRadius2, False)

    End Sub


    Sub CreateSegment(oCompDef As PartComponentDefinition, _
                     segmentIdx As Integer, _
                     r_inner As Double, r_outer As Double, _
                     ang_inner_deg As Double, ang_outer_deg As Double, _
                     extrude_height As Double, _
                     pole_height As Double, _
                     top_inner As Double, top_outer As Double)

        Dim oSketch As PlanarSketch
        Dim oProfile As Profile
        Dim oExtrudeDef As ExtrudeDefinition
        Dim oExtrude As ExtrudeFeature

        ' Create sketch on XY plane
        Set oSketch = oCompDef.Sketches.Add(oCompDef.WorkPlanes(3))
        oSketch.Name = "Seg_" & segmentIdx & "_Sketch"

        ' Convert angles to radians
        Dim ang_inner_rad As Double, ang_outer_rad As Double
        ang_inner_rad = ang_inner_deg * 3.14159265358979 / 180#
        ang_outer_rad = ang_outer_deg * 3.14159265358979 / 180#

        ' Create 2D polygon (annulus segment)
        Call DrawAnnulusSegment(oSketch, _
                               r_inner, r_outer, _
                               ang_inner_rad, ang_outer_rad)

        ' Create a profile
        Set oProfile = oSketch.Profiles.AddForSolid

        ' Create extrude definition
        Set oExtrudeDef = oCompDef.Features.ExtrudeFeatures.CreateExtrudeDefinition(oProfile, kNewBodyOperation)
        Call oExtrudeDef.SetDistanceExtent(extrude_height + pole_height, kPositiveExtentDirection)
        Set oExtrude = oCompDef.Features.ExtrudeFeatures.Add(oExtrudeDef)

        ' Apply chamfer to top edge if top surface is angled
        If Not IsClose(top_inner, top_outer) Then
            ' Determine which edge to chamfer
            Dim chamferRadius1 As Double  ' vertical distance
            Dim chamferRadius2 As Double  ' radial distance
            Dim shouldChamferInner As Boolean

            shouldChamferInner = (top_inner < top_outer)
            chamferRadius1 = Abs(top_inner - top_outer)
            chamferRadius2 = r_outer - r_inner

            ' Apply chamfer (note: radial distance first, then vertical)
            Call ApplyTopChamfer(oCompDef, oExtrude, chamferRadius2, chamferRadius1, shouldChamferInner)
        End If

    End Sub


    Sub CreatePoleSegments(oCompDef As PartComponentDefinition, _
                          innerRadius As Double, outerRadius As Double, _
                          poleHeight As Double, poleHalfAngleDeg As Double, _
                          nSegments As Integer, _
                          sideShims() As Double, topShims() As Double)

        Dim i As Integer
        Dim r_inner As Double, r_outer As Double
        Dim ang_inner_deg As Double, ang_outer_deg As Double
        Dim height As Double

        ' Loop through each radial segment
        For i = 0 To nSegments - 1

            ' Radii
            r_inner = innerRadius + (outerRadius - innerRadius) * i / nSegments
            r_outer = innerRadius + (outerRadius - innerRadius) * (i + 1) / nSegments

            ' Angular positions (absolute angles on pole surface)
            ang_inner_deg = poleHalfAngleDeg + sideShims(i)
            ang_outer_deg = poleHalfAngleDeg + sideShims(i + 1)

            ' Extrusion height is the maximum of the two adjacent top offsets
            height = IIf(topShims(i) > topShims(i + 1), topShims(i), topShims(i + 1))

            ' Create segment
            Call CreateSegment(oCompDef, i, _
                              r_inner, r_outer, ang_inner_deg, ang_outer_deg, _
                              height, poleHeight, topShims(i), topShims(i + 1))
        Next i

        ' ===== COMBINE ALL BODIES INTO ONE SOLID =====
        Dim oBodies As SurfaceBodies
        Dim oBaseBody As SurfaceBody
        Dim oToolBodies As ObjectCollection
        Set oToolBodies = ThisApplication.TransientObjects.CreateObjectCollection

        ' Collect all solid bodies
        Set oBodies = oCompDef.SurfaceBodies

        If oBodies.Count > 1 Then
            ' Create union of all bodies
            Dim j As Integer
            Dim oUnionFeature As CombineFeature

            Set oBaseBody = oBodies.Item(1)

            ' Union with remaining bodies
            For j = 2 To oBodies.Count
                Call oToolBodies.Add(oBodies.Item(j))
            Next j

            ' Perform union
            Set oUnionFeature = oCompDef.Features.CombineFeatures.Add(oBaseBody, oToolBodies, kJoinOperation, False)
        End If
    End Sub

    Sub CreateCyclotronPole()

        Dim oDoc As PartDocument
        Dim oCompDef As PartComponentDefinition

        ' Get active Inventor document
        On Error Resume Next
        Set oDoc = ThisApplication.ActiveDocument
        If oDoc Is Nothing Then
            MsgBox "No active document. Please create a new part first.", vbCritical
            Exit Sub
        End If

        On Error GoTo 0
        Set oCompDef = oDoc.ComponentDefinition

        ' Pole parameters (cm, degrees)
        Const INNER_RADIUS As Double = {0.1 * inner_rad}#
        Const OUTER_RADIUS As Double = {0.1 * outer_rad}#
        Const pole_height As Double = {0.1 * pole_height}#
        Const POLE_ANGLE_DEG As Double = {pole_angle_deg}#
        Const POLE_HALF_ANGLE_DEG As Double = {pole_half_angle_deg}
        Const N_SEGMENTS As Integer = {n_segments}

        ' Shim parameters
        Dim sideShims() As Double
        Dim topShims() As Double

        ReDim sideShims(0 To {n_segments})
        ReDim topShims(0 To {n_segments})

        ' Side shim offsets (degrees) - angular displacement at each radius
    {side_assignments}

        ' Top shim offsets (cm) - vertical displacement at each radius
    {top_assignments}

        ' Create pole segments
        Call CreatePoleSegments(oCompDef, INNER_RADIUS, OUTER_RADIUS, pole_height, _
                               POLE_HALF_ANGLE_DEG, N_SEGMENTS, sideShims, topShims)

        MsgBox "Cyclotron pole created successfully!" & vbCrLf & _
               "  Segments: " & N_SEGMENTS & vbCrLf & _
               "  Inner radius: " & INNER_RADIUS & " mm" & vbCrLf & _
               "  Outer radius: " & OUTER_RADIUS & " mm", vbInformation

    End Sub
    '''

        return macro
