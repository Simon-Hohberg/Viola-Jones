from PIL import Image
import numpy as np

'''
In an integral image each pixel is the sum of all pixels in the original image 
that are 'left and above' the pixel.

Original    Integral
+--------   +------------
| 1 2 3 .   | 0  0  0  0 .
| 4 5 6 .   | 0  1  3  6 .
| . . . .   | 0  5 12 21 .
            | . . . . . .

'''
class IntegralImage:

    def __init__(self, imageSrc, label):
        self.original = np.array(Image.open(imageSrc))
        self.sum = 0
        self.label = label
        self.calculate_integral()
        self.weight = 0
    
    def calculate_integral(self):
        # an index of -1 refers to the last row/column
        # since rowSum is calculated starting from (0,0),
        # rowSum(x, -1) == 0 holds for all x
        rowSum = np.zeros(self.original.shape)
        # we need an additional column and row
        self.integral = np.zeros((self.original.shape[0]+1, self.original.shape[1]+1))
        for x in range(self.original.shape[1]):
            for y in range(self.original.shape[0]):
                rowSum[y, x] = rowSum[y-1, x] + self.original[y, x]
                self.integral[y+1, x+1] = self.integral[y+1, x-1+1] + rowSum[y, x]
    
    def get_area_sum(self, topLeft, bottomRight):
        '''
        Calculates the sum in the rectangle specified by the given tuples.
        @param topLeft: (x,y) of the rectangle's top left corner
        @param bottomRight: (x,y) of the rectangle's bottom right corner 
        '''
        
        # swap tuples
        topLeft = (topLeft[1], topLeft[0])
        bottomRight = (bottomRight[1], bottomRight[0])
        if topLeft == bottomRight:
            return self.integral[topLeft]
        topRight = (bottomRight[0], topLeft[1])
        bottomLeft = (topLeft[0], bottomRight[1])
        return self.integral[bottomRight] - self.integral[topRight] - self.integral[bottomLeft] + self.integral[topLeft] 
    
    def set_label(self, label):
        self.label = label
    
    def set_weight(self, weight):
        self.weight = weight