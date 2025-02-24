import freetype
import bezier
import numpy as np
from matplotlib.path import Path as matPath

from svgpathtools import svg2paths, Path, Line, Arc, CubicBezier, QuadraticBezier

'''
Font parser.
'''
class BezierGlyph:

    def __init__(self, font_path):

        self.face = freetype.Face(font_path)
        self.outline = None
        self.bezier_list = []

    def load_character(self, char):
        self.bezier_list.clear()

        self.face.load_char(char)
        self.outline = self.face.glyph.outline

        points, codes = [], []
        def move_to(p, _):
            points.append((p.x, p.y))
            codes.append(matPath.MOVETO)
        def segment_to(*args):
            *args, _ = args
            points.extend([(p.x, p.y) for p in args])
            code = (matPath.LINETO, matPath.CURVE3, matPath.CURVE4)[len(args)-1]
            codes.extend([code] * len(args))
        self.outline.decompose(None, move_to, segment_to, segment_to, segment_to)

        prev_points = []
        code_len = len(codes)
        for i in range(code_len):
            curr_point = points[i]
            prev_points.append(curr_point)

            code = codes[i]

            if code == matPath.MOVETO:

                prev_points = prev_points[-1:]
                continue
            
            elif code == matPath.LINETO:
                assert len(prev_points) == 2, ""

                nodes = np.zeros((2, 2), dtype=np.float32)
                nodes[0] = np.asarray(prev_points[0], dtype=np.float32)
                nodes[1] = np.asarray(prev_points[1], dtype=np.float32)
                curve = bezier.Curve(nodes.transpose(), 1)
                self.bezier_list.append(curve)

                prev_points = prev_points[-1:]

            elif code == matPath.CURVE3:

                if len(prev_points) == 2:
                    continue
                elif len(prev_points) == 3:
                    nodes = np.zeros((3, 2), dtype=np.float32)
                    nodes[0] = np.asarray(prev_points[0], dtype=np.float32)
                    nodes[1] = np.asarray(prev_points[1], dtype=np.float32)
                    nodes[2] = np.asarray(prev_points[2], dtype=np.float32)
                    curve = bezier.Curve(nodes.transpose(), 2)
                    self.bezier_list.append(curve)

                    prev_points = prev_points[-1:]
                else:
                    raise ValueError()

            elif code == matPath.CURVE4:

                if len(prev_points) < 4:
                    continue
                elif len(prev_points) == 4:
                    nodes = np.zeros((4, 2), dtype=np.float32)
                    nodes[0] = np.asarray(prev_points[0], dtype=np.float32)
                    nodes[1] = np.asarray(prev_points[1], dtype=np.float32)
                    nodes[2] = np.asarray(prev_points[2], dtype=np.float32)
                    nodes[3] = np.asarray(prev_points[3], dtype=np.float32)
                    curve = bezier.Curve(nodes.transpose(), 3)
                    self.bezier_list.append(curve)

                    prev_points = prev_points[-1:]
                else:
                    raise ValueError()
                
            else:
                raise ValueError()
            
    def sample(self, num_sample_per_curve):

        num_curve = len(self.bezier_list)
        samples = np.zeros((num_curve * num_sample_per_curve, 2), dtype=np.float32)
        for i in range(num_curve):
            curve = self.bezier_list[i]
            # random_x = np.random.random((num_sample_per_curve))
            random_x = np.linspace(0., 1., num_sample_per_curve)
            curr_samples = curve.evaluate_multi(random_x).transpose()

            samples[num_sample_per_curve * i : num_sample_per_curve * (i + 1)] = curr_samples

        return samples
    
    def to_mesh(self, num_sample_per_curve):
        '''
        Change this font to mesh, which is a list of straight line segments.
        If a bezier curve segment was already a straight line, just select end points.
        Return:
        @ points: [# point, 2]
        @ edges: [# edge, 2]
        '''

        assert num_sample_per_curve > 1, ""

        num_curve = len(self.bezier_list)

        end_points = {}     # key: id of given end point in [points], value: end point coords
        points = []
        edges = []

        def find_end_point(epoint):
            for k, v in end_points.items():
                d = v - epoint
                d = np.linalg.norm(d, ord=2)
                if d < 1e-5:
                    return k
            return -1

        for i in range(num_curve):

            curve = self.bezier_list[i]

            nodes = curve.nodes.transpose()

            # process end points;

            beg = nodes[0]
            end = nodes[-1]

            beg_id = find_end_point(beg)
            end_id = find_end_point(end)

            if beg_id == -1:
                points.append(beg)
                end_points[len(points) - 1] = beg

            if end_id == -1:
                points.append(end)
                end_points[len(points) - 1] = end

            if curve.degree == 1:
                # just use end points;
                beg = nodes[0]
                end = nodes[-1]

                beg_id = find_end_point(beg)
                end_id = find_end_point(end)

                edge = []
                edge.append(beg_id)
                edge.append(end_id)
                edges.append(edge)

            else:
                random_x = np.linspace(0., 1., num_sample_per_curve)
                curr_samples = curve.evaluate_multi(random_x).transpose()

                for j in range(num_sample_per_curve - 1):
                    edge = []
                    cbeg = curr_samples[j]
                    cend = curr_samples[j + 1]

                    if j == 0:
                        beg_id = find_end_point(cbeg)
                    else:
                        beg_id = len(points) - 1

                    if j == num_sample_per_curve - 2:
                        end_id = find_end_point(cend)
                    else:
                        end_id = len(points)
                        points.append(cend)

                    edge.append(beg_id)
                    edge.append(end_id)
                    edges.append(edge)

        points = np.array(points)
        edges = np.array(edges)
        
        return points, edges

'''
SVG parser.
'''
def sample_points_on_path(path, num_points=100):
    points = []
    for i in range(num_points + 1):
        t = i / num_points
        pt = path.point(t)
        pt_x = pt.real
        pt_y = pt.imag
        points.append([pt_x, pt_y])
    return points

def sample_points_from_svg(svg_file, num_points_per_segment=100):
    paths, attributes = svg2paths(svg_file)
    all_sampled_points = []
    segment_points = []
    for path in paths:
        sampled_points = []
        for segment in path:
            curr_segment_points = sample_points_on_path(segment, num_points=num_points_per_segment)
            sampled_points.extend(curr_segment_points)
            segment_points.append(curr_segment_points)
        all_sampled_points.extend(sampled_points)
    return all_sampled_points, segment_points