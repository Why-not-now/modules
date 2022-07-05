import tkinter as tk
import numpy as np
import tools


class Frame(tk.Frame):
    """Illustrate how to _drag items on a Tkinter canvas"""

    def __init__(self, parent: tk.Tk) -> None:
        super().__init__(parent)

        self.width = parent.winfo_screenwidth()
        self.height = parent.winfo_screenheight()
        parent.geometry(f'{self.width}x{self.height}')
        self.origin = [self.width / 2, self.height / 2]
        self.globsize = 1

        # create a canvas
        self.canvas = tk.Canvas(width=self.width,
                                height=self.height,
                                background="bisque")
        self.canvas.pack(fill="both", expand=True)

        # this data is used to keep track of an
        # item being dragged
        self._drag_data = {"x": 0, "y": 0, "item": None, "press": False}
        self._widgets: dict[int, tuple] = {}

        # add bindings for clicking, dragging and releasing over
        # any object with the "token" tag
        self.canvas.tag_bind("token", "<ButtonPress-1>", self._drag_start)
        self.canvas.tag_bind("token", "<ButtonRelease-1>", self._drag_stop)
        self.canvas.tag_bind("token", "<B1-Motion>", self._drag)
        self.canvas.tag_bind("token", "<Enter>", self._select)
        self.canvas.tag_bind("token", "<Leave>", self._deselect)
        self.canvas.tag_bind("curve", "<B1-Motion>", self._update_curve)
        parent.bind("<Key-Return>", lambda _: parent.destroy())
        # self.canvas.bind("<MouseWheel>", self.zoom)

    def create_token(self, x, y, color, **kwargs):
        """Create a token at the given coordinate in the given color"""
        tags = ("token",) + kwargs.pop("tags", ())
        token = self.canvas.create_oval(
            x - 5,
            y - 5,
            x + 5,
            y + 5,
            outline=color,
            fill=color,
            tags=tags,
            **kwargs
        )
        self._widgets[token] = {}
        return token

    def _drag_start(self, event):
        """Begining _drag of an object"""
        # record the item and its location
        self._drag_data["press"] = True
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y

    def _drag_stop(self, event):
        """End _drag of an object"""
        # reset the _drag information
        self._drag_data["press"] = False
        self._drag_data["x"] = 0
        self._drag_data["y"] = 0

    def _drag(self, event):
        """Handle dragging of an object"""
        # compute how much the mouse has moved
        self._drag_data["delta_x"] = event.x - self._drag_data["x"]
        self._drag_data["delta_y"] = event.y - self._drag_data["y"]
        # move the object the appropriate amount
        self.canvas.move(
            self._drag_data["item"],
            self._drag_data["delta_x"], self._drag_data["delta_y"]
        )
        # record the new position
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y

    # def zoom(self, event):
    #     if (event.delta > 0):
    #         # if follow == True:
    #         #     self.canvas.scale("all", self.width/2, self.height/2,
    #         #                       1.1, 1.1)
    #         #     origin = [origin[0]*1.1 - self.width/2*0.1,
    #         #               origin[1]*1.1 - self.height/2*0.1]
    #         # else:
    #         self.canvas.scale("all", self.canvas.canvasx(event.x),
    #                           self.canvas.canvasy(event.y), 1.1, 1.1)
    #         origin = [origin[0]*1.1 - self.canvas.canvasx(event.x)*0.1,
    #                   origin[1]*1.1 - self.canvas.canvasy(event.y)*0.1]
    #         self.globsize *= 1.1
    #     elif (event.delta < 0):
    #         # if follow  == True:
    #         #     self.canvas.scale("all", self.width/2, self.height/2,
    #         #                       0.9, 0.9)
    #         #     origin = [origin[0]*0.9 + self.width/2*0.1,
    #         #               origin[1]*0.9 + self.height/2*0.1]
    #         # else:
    #         self.canvas.scale("all", self.canvas.canvasx(event.x),
    #                           self.canvas.canvasy(event.y), 0.9, 0.9)
    #         origin = [origin[0]*0.9 + self.canvas.canvasx(event.x)*0.1,
    #                   origin[1]*0.9 + self.canvas.canvasy(event.y)*0.1]
    #         self.globsize *= 0.9
    #     self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _select(self, event):
        if not self._drag_data["press"]:
            with_tag = self.canvas.find_withtag("token")
            x, y = event.x, event.y
            for widget in \
                    self.canvas.find_overlapping(x - 5, y - 5, x + 5, y + 5):
                if widget in with_tag:
                    self._drag_data["item"] = widget
                    break
            self.canvas.itemconfigure(
                self._drag_data["item"],
                outline="red",
                fill="red",
            )

    def _deselect(self, event):
        if not self._drag_data["press"]:
            self.canvas.itemconfigure(
                self._drag_data["item"],
                outline="green",
                fill="green",
            )
            self._drag_data["item"] = None

    def create_curve(self,
                     inputs=None,
                     func=None,
                     update_func=None,
                     cls=None,
                     tokens=None,
                     **kwargs):
        if tokens is None:
            tokens = tuple(self.create_token(*_, "green", tags=("curve",))
                           for _ in inputs)
        else:
            inputs = []
            for token in tokens:
                x_1, y_1, x_2, y_2 = self.canvas.coords(token)
                x = x_1 + 5
                y = y_1 + 5
                inputs.append([x, y])

        if cls is None:
            curve = self.canvas.create_line(func(inputs), **kwargs)
            instance = None
        else:
            instance = cls(inputs)
            curve = self.canvas.create_line(func(instance), **kwargs)
        self.canvas.tag_raise("token")

        if update_func is None:
            lst = [curve, func, False, cls]
        else:
            lst = [curve, update_func, True, instance]
        for index, token in enumerate(tokens):
            self._widgets[token][curve] = [index] + lst[1:]
        self._widgets[curve] = \
            [[_1] + _2 + lst[1:] for _1, _2 in zip(tokens, inputs)]

        return instance, curve, tokens

    def create_rectangle(self,
                         inputs=None,
                         func=None,
                         update_func=None,
                         cls=None,
                         tokens=None,
                         **kwargs):
        if tokens is None:
            tokens = tuple(self.create_token(*_, "green", tags=("curve",))
                           for _ in inputs)
        else:
            inputs = []
            for token in tokens:
                x_1, y_1, x_2, y_2 = self.canvas.coords(token)
                x = x_1 + 5
                y = y_1 + 5
                inputs.append([x, y])

        if cls is None:
            curve = self.canvas.create_rectangle(func(inputs), **kwargs)
            instance = None
        else:
            instance = cls(inputs)
            curve = self.canvas.create_rectangle(func(instance), **kwargs)
        self.canvas.tag_raise("token")

        if update_func is None:
            lst = [curve, func, False, cls]
        else:
            lst = [curve, update_func, True, instance]
        for index, token in enumerate(tokens):
            self._widgets[token][curve] = [index] + lst[1:]
        self._widgets[curve] = \
            [[_1] + _2 + lst[1:] for _1, _2 in zip(tokens, inputs)]

        return instance, curve, tokens

    def create_oval(self,
                    inputs=None,
                    func=None,
                    update_func=None,
                    cls=None,
                    tokens=None,
                    **kwargs):
        if tokens is None:
            tokens = tuple(self.create_token(*_, "green", tags=("curve",))
                           for _ in inputs)
        else:
            inputs = []
            for token in tokens:
                x_1, y_1, x_2, y_2 = self.canvas.coords(token)
                x = x_1 + 5
                y = y_1 + 5
                inputs.append([x, y])

        if cls is None:
            curve = self.canvas.create_oval(func(inputs), **kwargs)
            instance = None
        else:
            instance = cls(inputs)
            curve = self.canvas.create_oval(func(instance), **kwargs)
        self.canvas.tag_raise("token")

        if update_func is None:
            lst = [curve, func, False, cls]
        else:
            lst = [curve, update_func, True, instance]
        for index, token in enumerate(tokens):
            self._widgets[token][curve] = [index] + lst[1:]
        self._widgets[curve] = \
            [[_1] + _2 + lst[1:] for _1, _2 in zip(tokens, inputs)]

        return instance, curve, tokens

    def _update_curve(self, event):
        x_1, y_1, x_2, y_2 = self.canvas.coords(self._drag_data["item"])
        x = x_1 + 5
        y = y_1 + 5
        for curve, lst in self._widgets[self._drag_data["item"]].items():
            index, func, update, instance_cls = lst
            self._widgets[curve][index][1] = x
            self._widgets[curve][index][2] = y

            if update:
                if instance_cls is None:
                    args = index, [x, y]
                else:
                    args = instance_cls, index, [x, y]
            else:
                args = [[_[1], _[2]] for _ in self._widgets[curve]]
                if instance_cls is not None:
                    args = instance_cls(args)
                args = (args,)

            b = func(*args)
            a = tools.flatten(b)
            self.canvas.coords(
                curve,
                *tools.flatten(a)
            )


def _main():
    class line():
        def __init__(self, points) -> None:
            self.inputs = np.array(points, dtype=np.float64)

        def update(self, index, value):
            self.inputs[index] = value
            return self.curve()

        def curve(self):
            self.curvepoints = self.inputs
            return self.curvepoints

    root = tk.Tk()
    frame = Frame(root)
    frame.pack(fill="both", expand=True)

    # create a couple of movable objects
    frame.create_token(100, 100, "white")
    frame.create_token(200, 100, "black")

    def create_line(instance):
        return line.curve(instance).tolist()

    def update_line(instance, index, value):
        return line.update(instance, index, value).tolist()

    frame.create_curve(
        [[500, 500], [1000, 1000]], create_line, update_line, line
    )
    root.mainloop()


if __name__ == "__main__":
    _main()
