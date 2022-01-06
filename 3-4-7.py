from typing import Callable, NamedTuple
from manim import *

def position(R: float, r: float, a: float, theta: float):
    return np.array((
        (R - r)*np.cos(theta) + r*a*np.cos((R-r)*theta/r), # X
        (R - r)*np.sin(theta) - r*a*np.sin((R-r)*theta/r), # Y
        0.0 # Z
    ), dtype=float)

def find_best_a(R: float, r: float) -> float:
    from scipy import integrate, optimize

    def x_(theta: float, a: float):
        return (R - r)*(-np.sin(theta) - a*np.sin((R-r)*theta/r))
    def x__(theta: float, a: float):
        return -(R - r)*(np.cos(theta) + (R - r)*a*np.cos((R-r)*theta/r)/r)

    def y_(theta: float, a: float):
        return (R - r)*(np.cos(theta) - a*np.cos((R-r)*theta/r))
    def y__(theta: float, a: float):
        return -(R - r)*(np.sin(theta) - (R - r)*a*np.sin((R-r)*theta/r)/r)

    def k2(theta: float, a: float):
        return (x_(theta, a)*y__(theta, a) - x__(theta, a)*y_(theta, a))**2/(x_(theta, a)**2 + y_(theta, a)**2)**3
    
    # def s(theta: float, a: float):
    #     return 2*r*(R-r)*(1-a)*special.ellipeinc(R*theta/(2*r), -4*a/(1-a)**2)

    def f(a: float):
        res = integrate.quad(lambda t: (R-r)*k2(t, a)*np.sqrt(1 + a*a - 2*a*np.cos(R*t/r)), 2*PI*r/R*0.2, 2*PI*r/R*0.8)
        # print(res)
        return res[0]

    # f(.9)
    
    return optimize.minimize_scalar(f, bounds=(0.5, 1), method='bounded')

class SetupScene(Scene):
    def setup(self):
        class GlobalVars(NamedTuple):
            r: float
            N: int
            n: int
            a: float
            pos_func: Callable[[float], np.ndarray]
            pos: ParametricFunction

        r = 3
        N = int(input("Type the n-pointed star's edge count: "))
        if (N <= 2):
            raise ValueError("The value must be greater than 2.")
        n = int(input("Type the first shape's edge count: "))
        if (N <= n):
            raise ValueError("This number must be smaller than the previous one.")
        if (n < 1):
            raise ValueError("The value must be positive")
        if (np.gcd(N, n) != 1):
            raise ValueError("These numbers must be coprime. Try dividing them by their gcd.")

        a = find_best_a(r, r*n/N).x

        pos_func = lambda x: position(r, r*n/N, a, x)

        pos = ParametricFunction(pos_func, t_range = np.array([0, TAU*n]), fill_opacity=0)

        self.globalvars = GlobalVars(r, N, n, a, pos_func, pos)

class CreateFigure(SetupScene):
    def construct(self):
        # Sets up the relevant variables
        r, N, n, a, pos_func, pos = self.globalvars

        pos.set_color(RED).set_z_index(-1)

        # Creates the circles that are going to be drawn on the scene, as well as the inner circle's path
        biggerCircle = Circle(radius=r, color=WHITE, fill_opacity=0)
        smallerCircle = Circle(radius=r*n/N, color=WHITE, fill_opacity=0).shift(r*(1 - n/N)*RIGHT)
        circle_center_path = ParametricFunction(lambda x: r*(1 - n/N)*np.array((np.cos(3*x), np.sin(3*x), 0)), t_range = np.array((0, TAU)))

        # Creates two dots: d0 is the inner circle's center and d1 is the dot that's going to draw the heptagram
        d0 = Dot(r*(1 - n/N)*RIGHT, color=ORANGE)
        d1 = Dot(pos_func(0), color=RED) # pos_func(0) = (r*(1 - n/N*(1 - a)), 0, 0)

        # Creates the line that connects d0 and d1
        l1 = Line(d0.get_center(), d1.get_center(), color=ORANGE)
        l1_updater = lambda z: z.become(Line(d0.get_center(), d1.get_center(), color=ORANGE))
        l1.add_updater(l1_updater)

        # Plays the creation of all the aforementioned Mobjects
        self.play(LaggedStart(Create(biggerCircle), Create(smallerCircle), Create(d0), Create(l1), Create(d1), lag_ratio=.5), run_time=1)

        self.wait(1)

        # Plays the creation of the heptagram by rolling the inner circle around the outer circle
        func = rate_functions.ease_in_out_sine
        self.play(Create(pos, rate_func=func), 
            MoveAlongPath(d0, circle_center_path, rate_func=func),
            MoveAlongPath(d1, pos, rate_func=lambda _: 1),
            MoveAlongPath(smallerCircle, circle_center_path, rate_func=func),
            run_time=1.5*N)

        self.wait(.5)

        # Removes everything but the heptagram
        l1.remove_updater(l1_updater)
        self.play(FadeOut(d0), FadeOut(d1), FadeOut(l1), FadeOut(smallerCircle), FadeOut(biggerCircle))

        self.wait(3)

        # Removes the heptagram
        self.play(FadeOut(pos, run_time=2))

        self.wait(1)

class ShowMovingPoints(SetupScene):
    def construct(self):
        # Sets up the relevant variables
        r, N, n, a, pos_func, pos = self.globalvars

        pos.set_z_index(0).set_color(PURPLE)

        class AutoDot(Dot):
            theta: float = 0

            def __init__(self, start_val: float = 0, **kwargs):
                if 'point' in kwargs:
                    del kwargs['point']
                
                self.theta = start_val

                super().__init__(point=pos_func(start_val), **kwargs)

                def d_updater(z, dt):
                    self.theta += dt
                    z.become(AutoDot(start_val=self.theta, **kwargs))
                
                self.add_updater(d_updater)

        dot_list: list[AutoDot] = []

        # Creates the dots
        for i in range(N-n):
            for j in range(n):
                dot_list.append(AutoDot(n*TAU*i/(N-n) + TAU*j, color=RED).set_z_index(2))
        
        class AutoLine(Line):
            def __init__(self, start_dot: Dot, end_dot: Dot, **kwargs):
                if 'start' in kwargs:
                    del kwargs['start']
                if 'end' in kwargs:
                    del kwargs['end']

                super().__init__(start=start_dot.get_center(), end=end_dot.get_center(), **kwargs)

                self.add_updater(lambda z: z.become(AutoLine(start_dot=start_dot, end_dot=end_dot,stroke_opacity=z.get_stroke_opacity(), **kwargs)))
        
        def sorter(l: List[AutoDot]):
            g = VGroup(*l)

            x = g.get_x()

            y = g.get_y()

            return l.sort(key=lambda z: np.arctan2(z.get_y() - y, z.get_x() - x))

        triangle_list: list[list[Line]] = []
        # Creates the first shape
        for i in range(N-n):
            sublist = dot_list[n*i:n*(i+1)]

            sorter(sublist)

            vg = []
            triangle_list.append(vg)

            for j in range(n):
                line: AutoLine = AutoLine(sublist[j-1], sublist[j], color=ORANGE).set_z_index(1)
                vg.append(line)

        square_list: list[list[Line]] = []
        # Creates the second shape
        for i in range(n):
            sublist = dot_list[i::n]

            sorter(sublist)

            vg = []
            square_list.append(vg)

            for j in range(N-n):
                line: AutoLine = AutoLine(sublist[j-1], sublist[j], color=YELLOW).set_z_index(1)
                vg.append(line)

        self.add(*[d for d in dot_list])
        self.wait(N)

        # class FadeWhileMoving(Animation):
        #     mobject: VMobject
        #     isin: bool
        #     def __init__(self, mobject: VMobject, isin=True, **kwargs) -> None:
        #         self.isin = isin
        #         super().__init__(mobject, **kwargs)
        #     def interpolate_mobject(self, alpha: float) -> None:
        #         self.mobject.set_opacity(self.isin*(2*alpha - 1) + 1 - alpha)

        self.play(*[FadeIn(*v) for v in triangle_list])
        self.wait(N)
        self.play(*[FadeOut(*v) for v in triangle_list])

        self.wait(3)

        self.play(*[FadeIn(*v) for v in square_list])
        self.wait(N)
        self.play(*[FadeOut(*v) for v in square_list])

        self.wait(3)

        self.play(FadeIn(pos))
        self.wait(N)
        self.play(FadeOut(pos))

        self.wait(3)

        # self.play(AnimationGroup(FadeIn(pos), *[FadeIn(*v) for v in triangle_list + square_list]))
        # self.wait(N)
        # self.play(AnimationGroup(FadeOut(pos), *[FadeOut(*v) for v in triangle_list + square_list]))

        # self.wait(3)





# This is mostly for test purposes
def main():
    a = find_best_a(1, 2/11)
    print(a)

if __name__ == "__main__":
    main()