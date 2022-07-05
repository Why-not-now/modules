import numpy as np
import time
import tools


default_updates = ['curve',
                   'generate_distance_LUT']


class bezier:
    def __init__(self, points, resolution=1000, updates=None):

        if updates is None:
            updates = default_updates

        points = np.array(points, dtype=np.float64)
        tools.ValErr(2, len(points.shape), 'axis', 'axes')

        self.degree = points.shape[0] - 1
        self.dimension = points.shape[1]
        self.inputs = points
        self.resolution = resolution

        self.updates = updates
        self.update()

    def accept(self, value, start=0, end=1):
        if start <= value <= end:
            return True
        return False

    def update(self, *,
               points=None,
               point=None,
               add=None,
               remove=False,
               position=None):
        if points is not None:
            self.inputs = points
        elif point is not None:
            self.inputs[position] = point
        elif add is not None:
            self.inputs = np.insert(self.inputs, position, add)
        elif remove:
            self.inputs = np.delete(self.inputs, position)

        t1 = time.perf_counter_ns()
        retlst = {}

        for update in self.updates:
            retlst[update] = getattr(self, update)()

        t2 = time.perf_counter_ns()
        retlst['time'] = (t2 - t1) * 1e9
        return retlst

    def point(self, t):
        if t == 0:
            return self.inputs[0]
        if t == 1:
            return self.inputs[-1]

        mt = 1 - t
        sum = np.zeros(self.dimension)
        step = t / mt
        coeff = mt**self.degree
        for index, point in enumerate(self.inputs):
            sum += point * coeff
            coeff *= step * (self.degree - index) / (index + 1)
        return sum.tolist()

    def curve(self, resolution=None, update=True):
        points = []
        if resolution is None:
            resolution = self.resolution

        t = 0
        curr = 1
        inputs = []
        for index, point in enumerate(self.inputs):
            inputs.append(point * curr)
            curr *= (self.degree - index) / (index + 1)

        points = [self.inputs[0].tolist()]
        for t in np.linspace(0, 1, resolution, endpoint=False)[1:]:
            mt = 1 - t
            sum = np.zeros(self.dimension)
            step = t / mt
            coeff = mt**self.degree
            for point in inputs:
                sum += point * coeff
                coeff *= step
            points.append(sum.tolist())
        points.append(self.inputs[-1].tolist())

        if update is True:
            self.curvepoints = points
        return points

    def split(self, t):
        left = []
        right = []
        mt = 1 - t
        start = 1
        step = t / mt

        for point in range(1, self.degree + 1):
            curr = start

            leftsum = curr * self.inputs[0]
            rightsum = curr * self.inputs[-point]
            for index, (leftpoint, rightpoint) in \
                    enumerate(zip(self.inputs[1:point],
                                  self.inputs[-point + 1:])):

                curr *= (point - index - 1) / (index + 1) * step
                leftsum += curr * leftpoint
                rightsum += curr * rightpoint
            left.append(leftsum)
            right.append(rightsum)
            start *= mt

        curr = start
        bothsum = curr * self.inputs[0]
        for index, point in enumerate(self.inputs[1:]):
            curr *= (self.degree - index) * step / (index + 1)
            bothsum += curr * point
        left.append(bothsum)
        right.append(bothsum)

        return (bezier(left), bezier(right))

    def lower_degree(self):
        M = np.array([[0] * i
                     + [(self.degree - i) / self.degree, (i + 1) / self.degree]
                     + [0] * (self.degree - i - 1)
                     for i in range(self.degree)])
        Matrix_multiplier = np.linalg.inv(M @ M.T) @ M
        points = Matrix_multiplier @ self.inputs
        return bezier(points)

    def raise_degree(self):
        points = [self.inputs[0]]
        step = 1 / (self.degree + 1)
        c = step

        for (point_0, point_1) in \
                zip(self.inputs[:-1], self.inputs[1:]):
            points.append(c * point_0 + (1 - c) * point_1)
            c += step
        points.append(self.inputs[-1])
        return bezier(points)

    def differentiate(self):
        if self.degree == 0:
            return bezier([[0] * self.dimension])
        return bezier(self.degree * np.diff(self.inputs, axis=0))

    def components(self):
        return tuple(bezier([axis]) for axis in self.inputs.T)

    def component(self, axis):
        return bezier(self.inputs[:, axis])

    def bound(self, threshold=1e-5):
        bounding_box = []

        for index, axis in enumerate(self.inputs.T):
            points = [axis[0], axis[-1]]
            multiple = [1, *[0] * self.degree]
            coefficients = []
            coefficient = 1

            for degree in range(1, self.degree + 1):
                multiple = [b - a for a, b in
                            zip(multiple + [0], [0] + multiple)][:-1]
                coefficients.append(coefficient * (axis * multiple).sum())
                coefficient *= self.degree / degree - 1

            roots = np.polynomial.Polynomial(coefficients).roots()

            for root in roots.real[roots.imag < threshold]:
                if self.accept(root):
                    points.append(self.point(root)[index])

            bounding_box.append((min(points), max(points)))
        return tuple(zip(*bounding_box))

    # def align(self):    # TODO
    #     difference = self.inputs[-1] - self.inputs[0]
    #     difference

    def generate_distance_LUT(self, resolution=None, update: bool = True):
        if resolution is None:
            curve = np.array(self.curvepoints[1:], np.float64)
        else:
            curve = np.array(self.curve(resolution), np.float64)

        LUT = np.cumsum(np.linalg.norm(curve[:-1] - curve[1:], axis=1))

        if update is True:
            self.distance_LUT = LUT
        return LUT

    def distance_to_time(self, distance):
        if distance < 0:
            raise ValueError(f"Value {distance} is negative")
        if distance > self.distance_LUT[-1]:
            raise ValueError(f"Value {distance} is larger than \
the distance of curve {self.distance_LUT[-1]}")

        low = 0
        high = len(self.distance_LUT) - 1
        mid = high // 2

        while low != high:
            if self.distance_LUT[mid] > distance:
                high = mid
            else:
                low = mid + 1
            mid = low + (high - low) // 2

        bottom = self.distance_LUT[mid - 1]
        t_diff = self.distance_LUT[mid] - bottom
        d_diff = distance - bottom

        return (mid + d_diff / t_diff - 1) / (len(self.distance_LUT) - 1)

    def position_to_t(self, position, axis, threshold=1e-5):
        multiple = [1, *[0] * self.degree]
        axis = self.inputs.T[axis]
        coefficients = [axis[0] - position]
        coefficient = 1

        for degree in range(1, self.degree + 1):
            multiple = [b - a for a, b in
                        zip(multiple + [0], [0] + multiple)][:-1]
            coefficient *= (self.degree + 1) / degree - 1
            coefficients.append(coefficient * (multiple * axis).sum())

        roots = np.polynomial.Polynomial(coefficients).roots()

        return roots.real[roots.imag < threshold]


class quadratic_bezier(bezier):
    def __init__(self, points, resolution=1000, updates=None):
        super().__init__(points, resolution, updates)
        tools.ValErr(self.degree + 1, 3, 'point', 'points')

    def raise_degree(self):
        return cubic_bezier(super().raise_degree().inputs)

    def components(self):
        return (quadratic_bezier([axis]) for axis in self.inputs.T)

    def component(self, axis):
        return quadratic_bezier(self.inputs[:, axis])

    def bound(self, treshold=None):
        bounding_box = []
        for index, axis in enumerate(self.inputs.T):
            points = [axis[0], axis[-1]]

            roots = tools.linear_solve(
                ([1, -2, 1] * axis).sum(),
                ([-1, 1, 0] * axis).sum()
            )

            for root in roots:
                if self.accept(root):
                    points.append(self.point(root)[index])

            bounding_box.append((min(points), max(points)))
        return tuple(zip(*bounding_box))

    def position_to_t(self, position, axis, treshold=None):
        axis = self.inputs.T[axis]
        return tools.quadratic_solve(
            ([1, -2, 1] * axis).sum(),
            ([-1, 1, 0] * axis).sum(),
            ([1, 0, 0] * axis).sum() - position,
            np.linalg.norm(axis)
        )


class cubic_bezier(bezier):
    def __init__(self, points, resolution=1000, updates=None):
        super().__init__(points, resolution, updates)
        tools.ValErr(self.degree + 1, 4, 'point', 'points')

    def lower_degree(self):
        return quadratic_bezier(super().lower_degree().inputs)

    def raise_degree(self):
        return quartic_bezier(super().raise_degree().inputs)

    def components(self):
        return (cubic_bezier([axis]) for axis in self.inputs.T)

    def component(self, axis):
        return cubic_bezier(self.inputs[:, axis])

    def bound(self, treshold=None):
        bounding_box = []
        for index, axis in enumerate(self.inputs.T):
            points = [axis[0], axis[-1]]

            roots = tools.quadratic_solve(
                ([-1, 3, -3, 1] * axis).sum(),
                ([1, -2, 1, 0] * axis).sum(),
                ([-1, 1, 0, 0] * axis).sum(),
                np.linalg.norm(axis)
            )

            for root in roots:
                if self.accept(root):
                    points.append(self.point(root)[index])

            bounding_box.append((min(points), max(points)))
        return tuple(zip(*bounding_box))

    def position_to_t(self, position, axis, treshold=None):
        axis = self.inputs.T[axis]
        return tools.cubic_solve(
            ([-1, 3, -3, 1] * axis).sum(),
            ([1, -2, 1, 0] * axis).sum(),
            ([-1, 1, 0, 0] * axis).sum(),
            ([1, 0, 0, 0] * axis).sum() - position,
            np.linalg.norm(axis)
        )


class quartic_bezier(bezier):
    def __init__(self, points, resolution=1000, updates=None):
        super().__init__(points, resolution, updates)
        tools.ValErr(self.degree + 1, 5, 'point', 'points')

    def lower_degree(self):
        return cubic_bezier(super().lower_degree().inputs)

    def raise_degree(self):
        return quintic_bezier(super().raise_degree().inputs)

    def components(self):
        return (quartic_bezier([axis]) for axis in self.inputs.T)

    def component(self, axis):
        return quartic_bezier(self.inputs[:, axis])

    def bound(self, treshold=None):
        bounding_box = []
        for index, axis in enumerate(self.inputs.T):
            points = [axis[0], axis[-1]]

            roots = tools.cubic_solve(
                ([1, -4, 6, -4, 1] * axis).sum(),
                ([-1, 3, -3, 1, 0] * axis).sum(),
                ([1, -2, 1, 0, 0] * axis).sum(),
                ([-1, 1, 0, 0, 0] * axis).sum(),
                np.linalg.norm(axis)
            )

            for root in roots:
                if self.accept(root):
                    points.append(self.point(root)[index])

            bounding_box.append((min(points), max(points)))
        return tuple(zip(*bounding_box))

    def position_to_t(self, position, axis, treshold=None):
        axis = self.inputs.T[axis]
        return tools.quartic_solve(
            ([-1, 5, -10, 10, -5, 1] * axis).sum(),
            ([1, -4, 6, -4, 1, 0] * axis).sum(),
            ([-1, 3, -3, 1, 0, 0] * axis).sum(),
            ([1, -2, 1, 0, 0, 0] * axis).sum(),
            ([-1, 1, 0, 0, 0, 0] * axis).sum() - position,
            np.linalg.norm(axis)
        )


class quintic_bezier(bezier):
    def __init__(self, points, resolution=1000, updates=None):
        super().__init__(points, resolution, updates)
        tools.ValErr(self.degree + 1, 6, 'point', 'points')

    def lower_degree(self):
        return quartic_bezier(super().lower_degree().inputs)

    def components(self):
        return (quintic_bezier([axis]) for axis in self.inputs.T)

    def component(self, axis):
        return quintic_bezier(self.inputs[:, axis])

    def bound(self, treshold=None):  # TODO
        bounding_box = []
        for index, axis in enumerate(self.inputs.T):
            points = [axis[0], axis[-1]]

            roots = tools.quartic_solve(([-1, 5, -10, 10, -5, 1] * axis).sum(),
                                        ([1, -4, 6, -4, 1, 0] * axis).sum(),
                                        ([-1, 3, -3, 1, 0, 0] * axis).sum(),
                                        ([1, -2, 1, 0, 0, 0] * axis).sum(),
                                        ([-1, 1, 0, 0, 0, 0] * axis).sum(),
                                        np.linalg.norm(axis))

            for root in roots:
                if self.accept(root):
                    points.append(self.point(root)[index])

            bounding_box.append((min(points), max(points)))
        return tuple(zip(*bounding_box))


def create_bezier(points, *args, **kwargs):
    points = np.array(points, dtype=np.float64)
    return {
        2: quadratic_bezier,
        3: cubic_bezier,
        4: quartic_bezier,
        5: quintic_bezier
    }.get(points.shape[0], bezier)(points, *args, **kwargs)


def _main():
    import cv2 as cv
    import tkinter as tk
    from tkinter import ttk
    import cProfile
    import pstats
    import tkinter_tools

    root = tk.Tk()
    root.geometry(f'{root.winfo_screenwidth()}x{root.winfo_screenheight()}')
    print(f'{root.winfo_screenwidth()}x{root.winfo_screenheight()}')
    origin = [root.winfo_screenwidth() / 2, root.winfo_screenheight() / 2]
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()

    def bezier_curve(instance: bezier):
        return instance.curvepoints

    def update_bezier(instance: bezier, index, value):
        quintic_bezier.update(instance, point=value, position=index)
        return instance.curvepoints

    def create_skeleton(inputs):
        return inputs

    def create_bound(inputs):
        return curve.bound()

    def create_points(distance):
        def closure(inputs):
            t = curve.distance_to_time(distance * curve.distance_LUT[-1])
            x, y = curve.point(t)
            return x - 5, y - 5, x + 5, y + 5
        return closure

    with cProfile.Profile() as pr:
        frame = tkinter_tools.Frame(root)
        canvas = frame.canvas

        # bezier_0 = cubic_bezier([[100, 500],
        #                          [1500, 100],
        #                          [800, 700],
        #                          [300, 800]])
        # bezier_0 = quartic_bezier([[100, 500],
        #                           [1500, 100],
        #                           [800, 700],
        #                           [300, 800],
        #                           [900, 800]])
        # bezier_0 = quintic_bezier([[100, 500],
        #                            [1000, 400],
        #                            [300, 900],
        #                            [900, 800],
        #                            [1700, 300],
        #                            [1200, 100]])
        # bezier_0 = bezier([[100, 500], [1500, 100], [800, 700], [300, 800]])
        bezier_0, bezier_curve, tokens = frame.create_curve(
            [[100, 500],
             [1000, 400],
             [300, 900],
             [900, 800],
             [1700, 300],
             [1200, 100]],
            bezier_curve,
            update_bezier,
            create_bezier
        )
        curve = bezier_0
        skeleton_curve = frame.create_curve(
            func=create_skeleton,
            tokens=tokens
        )[1]
        bound_rectangle = frame.create_rectangle(
            func=create_bound,
            tokens=tokens,
            outline="#0F0"
        )[1]
        [
            frame.create_oval(
                func=create_points(y),
                tokens=tokens
            ) for y in np.linspace(0, 1, 101)]
        canvas.tag_lower(bound_rectangle)
        canvas.tag_raise(bound_rectangle, skeleton_curve)
        canvas.tag_raise(bezier_curve)
        canvas.tag_raise("token")
        # canvas.create_line(tools.flatten(
        #                    [(i/10, x/2) for i, x in
        #                     enumerate(curve.generate_distance_LUT())]))\,
        #                    fill="#000", smooth="False")
        # print(curve.distance_LUT[-1])
        # print(t)
    stats = pstats.Stats(pr)
    # stats.sort_stats(pstats.SortKey.TIME)
    # stats.print_stats()
    stats.dump_stats(filename='bezier.prof')

    root.mainloop()


if __name__ == "__main__":
    _main()
