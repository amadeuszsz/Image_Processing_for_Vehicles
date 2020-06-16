import numpy as np
from sign import Sign

class DetectedObjects():
    def __init__(self, labels, coords, width, height, min_obj_size=100):
        self.labels = labels
        self.coords = coords
        self.width = width
        self.height = height
        self.objects = np.unique(labels)
        self.objects_coords = []
        self.objects_coords_to_delete = []
        self.objects_coords_to_append = []
        self.signs = []
        self.min_obj_size = min_obj_size
        self.select_objects()
        self.getting_coords()
        self.finding_intersecting_rectangles()
        self.redefinition_rectangles_coords()

    def select_objects(self):
        self.objects = np.delete(self.objects, np.where(self.objects == 0))
        for object in self.objects:
            if np.count_nonzero(self.labels == object) < self.min_obj_size:
                self.labels[self.labels == object] = 0
                self.objects = np.delete(self.objects, np.where(self.objects == object))

    def getting_coords(self):
        for object in self.objects:
            most_left = self.width
            most_right = 0
            most_top = self.height
            most_bottom = 0
            for coord in self.coords:
                if self.labels[coord[0], coord[1]] == object:
                    if (coord[1] < most_left): most_left = coord[1]
                    if (coord[1] > most_right): most_right = coord[1]
                    if (coord[0] < most_top): most_top = coord[0]
                    if (coord[0] > most_bottom): most_bottom = coord[0]
            obj_coord = ObjectCoords(most_left, most_right, most_top, most_bottom)
            self.objects_coords.append(obj_coord)

    def finding_intersecting_rectangles(self):
        merging_flag = True
        while merging_flag:
            merging_flag = False
            for coord1 in self.objects_coords:
                for coord2 in self.objects_coords:
                    if (coord1 != coord2):
                        #Checking if any corner of 1st rectangle is inside 2nd rectangle
                        #Top left corner of coord2 inside coord1
                        if((coord2.most_left >= coord1.most_left and coord2.most_left <= coord1.most_right
                        and coord2.most_top >= coord1.most_top and coord2.most_top <= coord1.most_bottom)
                        #Top right corner of coord2 inside coord1
                        or (coord2.most_right >= coord1.most_left and coord2.most_right <= coord1.most_right
                        and coord2.most_top >= coord1.most_top and coord2.most_top <= coord1.most_bottom)
                        #Bottom left corner of coord2 inside coord1
                        or (coord2.most_left >= coord1.most_left and coord2.most_left <= coord1.most_right
                        and coord2.most_bottom >= coord1.most_top and coord2.most_bottom <= coord1.most_bottom)
                        #Bottom right corner of coord2 inside coord1
                        or (coord2.most_right >= coord1.most_left and coord2.most_right <= coord1.most_right
                        and coord2.most_bottom >= coord1.most_top and coord2.most_bottom <= coord1.most_bottom)
                        ):
                            # Appending intersecting rectangles to the delete list if they arent there yet
                            if(not coord1 in self.objects_coords_to_delete and  not coord2 in self.objects_coords_to_delete):
                                # Checking if rectangle exist in initial objects list and not exist delete list
                                if(coord1 in self.objects_coords and coord1 not in self.objects_coords_to_delete):
                                    self.objects_coords_to_delete.append(coord1)
                                    merging_flag = True
                                if(coord2 in self.objects_coords and coord2 not in self.objects_coords_to_delete):
                                    self.objects_coords_to_delete.append(coord2)
                                    merging_flag = True

                            # Creating a new large rectangle by connecting coord1 and coord2 and adding it to appending list
                            new_most_left = min(coord1.most_left, coord2.most_left)
                            new_most_right = max(coord1.most_right, coord2.most_right)
                            new_most_top = min(coord1.most_top, coord2.most_top)
                            new_most_bottom = max(coord1.most_bottom, coord2.most_bottom)
                            obj_coord_new = ObjectCoords(most_left=new_most_left,
                                                     most_right=new_most_right,
                                                     most_top=new_most_top,
                                                     most_bottom=new_most_bottom)
                            if(obj_coord_new not in self.objects_coords_to_append):
                                self.objects_coords_to_append.append(obj_coord_new)

                            # CHecking if merged rectangles already exists in append list
                            if(coord1 in self.objects_coords_to_append):
                                self.objects_coords_to_append.remove(coord1)
                            if (coord2 in self.objects_coords_to_append):
                                self.objects_coords_to_append.remove(coord2)

                            #self.objects_coords_to_delete = np.unique(self.objects_coords_to_delete)
                            #self.objects_coords_to_append = np.unique(self.objects_coords_to_append)

                            print("\nRectangles merged.\nRectangle 1 -> Left top: {0}x{1}, Right bottom: {2}x{3}\n"
                                  "Rectangle 2 -> Left top: {4}x{5}, Right bottom: {6}x{7}\n"
                                  "Rectangle N -> Left top: {8}x{9}, Right bottom: {10}x{11}"
                                  .format(coord1.most_left, coord1.most_top, coord1.most_right, coord1.most_bottom,
                                          coord2.most_left, coord2.most_top, coord2.most_right, coord2.most_bottom,
                                          new_most_left, new_most_top, new_most_right, new_most_bottom))
                            #print("Merging flag in a loop: ", merging_flag)
            #print("Merging flag at the end: ", merging_flag)
            if not merging_flag:
                break


    def redefinition_rectangles_coords(self):
        for rectangle in self.objects_coords_to_delete:
            self.objects_coords.remove(rectangle)
        for rectangle in self.objects_coords_to_append:
            self.objects_coords.append(rectangle)

        for obj_coord in self.objects_coords:
            sign = Sign(x=obj_coord.most_left, y=obj_coord.most_top, width=obj_coord.most_right - obj_coord.most_left,
                        height=obj_coord.most_bottom - obj_coord.most_top)
            self.signs.append(sign)

class ObjectCoords():
    def __init__(self, most_left, most_right, most_top, most_bottom):
        self.most_left = most_left
        self.most_right = most_right
        self.most_top = most_top
        self.most_bottom = most_bottom



