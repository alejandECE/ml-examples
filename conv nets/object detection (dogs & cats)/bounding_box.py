import matplotlib.patches as patches
import matplotlib.pyplot as plt


class BoundingBox:
  """
  Helper class to manage bounding boxes.
  """
  def __init__(self, x, y, width, height):
    self.x = x
    self.y = y
    self.width = width
    self.height = height

  def to_rectangle(self, color):
    """
    Converts bounding box to a matplotlib.patch rectangle.

    :param color: Edgecolor of the rectangle
    """
    return patches.Rectangle((self.x, self.y), self.width, self.height,
                             edgecolor=color,
                             facecolor='none',
                             lw=2)

  def intersect(self, other):
    """
    Returns the area of the intersection between both rectangles.

    :param other: The rectangle to compute the interception with.
    :return: Area of the interception.
    """
    dy = min(self.y + self.height, other.y + other.height) - max(self.y, other.y)
    if dy < 0:
      return 0
    dx = min(self.x + self.width, other.x + other.width) - max(self.x, other.x)
    if dx < 0:
      return 0
    return dx * dy

  def union(self, other):
    """
    Returns the area of the union between both rectangles.

    :param other: The rectangle to compute the union with.
    :return: Area of the union.
    """
    return self.width * self.height + other.width * other.height - self.intersect(other)

  def intersect_over_union(self, other):
    """
    Returns the interception over union (IoU) ratio between both rectangles.

    :param other: The rectangle to compute the IoU with.
    :return: IoU ratio.
    """
    return self.intersect(other) / self.union(other)


if __name__ == '__main__':
  # Creates two boxes
  box1 = BoundingBox(10, 5, 100, 200)
  box2 = BoundingBox(80, 150, 50, 100)
  # Plots both boxes
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.add_patch(box1.to_rectangle('r'))
  ax.add_patch(box2.to_rectangle('b'))
  plt.xlim([0, 500])
  plt.ylim([0, 500])
  plt.show()
  ax.invert_yaxis()
  # Computes IoU
  print('Intersection: ', box1.intersect(box2))
  print('Union: ', box1.union(box2))
  print('IoU: ', box1.intersect_over_union(box2))


