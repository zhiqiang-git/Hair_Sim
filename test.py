import taichi as ti
import sys
import argparse
import xml.etree.ElementTree as ET

ti.init(arch=ti.cuda)


a = ti.Vector([0, 1, 2])
b = ti.Vector([3, 4, 5])

c = a.outer_product(b)
print(c)