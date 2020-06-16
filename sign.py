
class Sign():
    def __init__(self,x,y,width,height):
        self.width = width
        self.height = height
        self.x = x
        self.y = y
        self.type = None # From template matching

    def print_info(self, key):
        print("Sign {4}-> size: {0}x{1}, coord: {2}x{3}".format(self.width, self.height, self.x, self.y, key))

    def corners_coords(self):
        top_left = [self.x, self.y]
        top_right = [self.x + self.width, self.y]
        bottom_left = [self.x, self.y + self.height]
        bottom_right = [self.x + self.width, self.y + self.height]

        return top_left, top_right, bottom_left, bottom_right

