from collections.abc import Iterable
import numpy as np


def flatten_generator(lst):
    for elst in lst:
        if isinstance(elst, Iterable) and \
                not isinstance(elst, (str, bytes)):
            yield from flatten_generator(elst)
        else:
            yield elst


def flatten(lst):
    return list(flatten_generator(lst))


def linear_solve(a, b):
    if a != 0:
        return [-b / a]

    else:
        return []


def quadratic_solve(a, b, c, avg, fit=True):
    if not fit:
        b /= 2
    if np.isclose(avg, avg + a, atol=0):
        return linear_solve(2 * b, c)

    elif np.isclose(avg, avg + (discriminant := b**2 - a * c)):
        return[-b / a]

    elif discriminant > 0:
        average = -b / a
        discriminant = np.sqrt(discriminant) / a
        return [average + discriminant, average - discriminant]

    else:
        return []


def cubic_solve(a, b, c, d, avg, fit=True):
    if not fit:
        b /= 3
        c /= 3
    if np.isclose(avg, avg + a, atol=0):
        return quadratic_solve(3 * b, 1.5 * c, d, avg)

    else:  # cubic
        b /= a
        c /= a
        d /= a
        avg = max(1, b, c, d)

        resultant_0 = b**2 - c
        resultant_1 = 2 * b**3 - 3 * b * c + d

        if (discriminant := resultant_1**2 - 4 * (resultant_0**3)) < 0:
            sqrt_resultant_0 = 2 * np.sqrt(resultant_0)  # 3 real roots
            phi = np.arccos(np.clip(-resultant_1
                            / (resultant_0 * sqrt_resultant_0), -1, 1))
            roots = [(sqrt_resultant_0 * np.cos(phi / 3)),
                     (sqrt_resultant_0 * np.cos((phi + 2 * np.pi) / 3)),
                     (sqrt_resultant_0 * np.cos((phi + 4 * np.pi) / 3))]

        elif discriminant > 0:  # 1 real root
            if resultant_0 == 0:
                roots = [-np.cbrt(resultant_1)]
            else:
                C = np.cbrt(0.5 * (resultant_1 + np.sqrt(discriminant)))
                roots = [-C - resultant_0 / C]

        elif resultant_0 == resultant_1 == 0:  # cubic expression
            roots = [0]

        else:  # 2 real root
            C = np.cbrt(0.5 * resultant_1)
            roots = [-2 * C, C]

        return [root - b for root in roots]


def quartic_solve(a, b, c, d, e, avg, fit=True):
    if not fit:
        b /= 4
        c /= 6
        d /= 4
    if np.isclose(avg, avg + a, atol=0):
        return cubic_solve(4 * b, 2 * c, 1.3333333333333333 * d, e, avg)

    else:  # quartic
        roots = []
        b /= a
        c /= a
        d /= a
        e /= a
        p = c - b**2
        q = 2 * b**3 - 3 * b * c + d
        s = e - 3 * b**4 + 6 * b**2 * c - 4 * b * d
        avg = max(1, p, q, s)

        if q == 0:  # biquadratic
            roots_quad = quadratic_solve(1, 3 * p, s, avg)
            for root in roots_quad:
                if np.isclose(avg, avg + root, atol=0):
                    roots.append(0)

                if root > 0:
                    roots.append(root := np.sqrt(root))
                    roots.append(-root)

        else:
            resultant_0 = 3 * p**2 + s
            resultant_1 = p**3 + q**2 - p * s
            if (np.isclose(
                avg, avg + (discriminant
                            := 27 * resultant_1**2 - resultant_0**3)
            )):

                if resultant_0 == 0:  # triple roots
                    roots.append(root := 0.25 * (5 * p**2 - s) / q)
                    roots.append(-3 * root)

                else:  # double roots
                    ps = p * s
                    p3 = 3 * p**3
                    q2 = 3 * q**2
                    roots.append(
                        root := s * (ps - 3 * p3 - q2)
                        / (3 * q * (2 * p3 + q2 - 2 * ps))
                    )
                    roots.extend(
                        quadratic_solve(1, root, 6 * p + 3 * root**2, avg)
                    )

            if discriminant > 0:  # 2 real root
                discriminant = np.sqrt(3 * discriminant)
                Q = np.cbrt(27 * resultant_1 + 3 * discriminant)
                S = np.sqrt(-p + Q / 6 + 0.5 * resultant_0 / Q)

            elif p < 0 and s - 9 * p**2 < 0:  # 4 real roots
                multiple = np.sqrt(resultant_0 / 3)
                phi = np.arccos(np.clip(
                    3 * resultant_1 / (resultant_0 * multiple), -1, 1))
                S = np.sqrt(-p + multiple * np.cos(phi / 3))

            else:  # 0 real root
                return []

            average = -S**2 - 3 * p
            difference = q / S
            if np.isclose(avg, avg + (discriminant := average + difference)):
                roots.append(-S)
            elif discriminant > 0:
                discriminant = np.sqrt(discriminant)
                roots.append(-S - discriminant)
                roots.append(-S + discriminant)
            if np.isclose(avg, avg + (discriminant := average - difference)):
                roots.append(S)
            elif discriminant > 0:
                discriminant = np.sqrt(discriminant)
                roots.append(S - discriminant)
                roots.append(S + discriminant)

        return [root - b for root in roots]


def ValErr(value, expected, singular, plural):
    if value != expected:
        if value == 1:
            string = f'Expected {value} {singular},'
        else:
            string = f'Expected {value} {plural},'
        if expected == 1:
            string += f' got {expected} {singular} instead'
        else:
            string += f' got {expected} {plural} instead'
        raise ValueError(string)
