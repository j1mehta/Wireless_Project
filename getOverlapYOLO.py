import shapely.geometry as sg
import ast
from shapely.geometry import Polygon
from shapely.ops import cascaded_union


 
def getRect(reg1):
    maxx, maxy, miny, minx = reg1['right'], reg1['bottom'], reg1['top'], reg1['left']
    return sg.Polygon([(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny),(minx, miny)])

#new_shape1 = cascaded_union([r1,r2,r3])
#new_shape2 = cascaded_union([r4,r5,r6])

#k = new_shape1.intersection(r7)
#print(k.area/new_shape1.area)


path500  = '/Users/jatin/Desktop/mu500PersonOnly.txt'
pathRef  = '/Users/jatin/Desktop/refPersonOnly.txt'
path200  = '/Users/jatin/Desktop/mu200PersonOnly.txt'
f500 = open('/Users/jatin/Desktop/mu500overlapScore.txt','w')
f200 = open('/Users/jatin/Desktop/mu200overlapScore.txt','w')

def getBlocks(path):
    blocks = []
    block = []
    f =  open(path)
    
    for line in f:
        if line[:3] in ('***'):
                blocks.append(block)
                block = []
        else:
            block.append(line.rstrip('\n'))
    if block:
        blocks.append(block)
    return blocks

blockRef = getBlocks(pathRef)
block500 = getBlocks(path500)
block200 = getBlocks(path200)

def getRectUnions(block_list):
    union_list = []
    for block in block_list:
        list_rect = []
        for rect in block:
            list_rect.append(getRect(ast.literal_eval(rect)))
        union_list.append(cascaded_union(list_rect))
    return union_list

ref_union = getRectUnions(blockRef)

union_500 = getRectUnions(block500)
union_200 = getRectUnions(block200)
k = zip(ref_union, union_500)
kk = zip(ref_union, union_200)

for i in k:
    int_area = i[0].intersection(i[1])
    overlap = int_area.area/i[0].area
    f500.write(str(overlap)+'\n')
    
for i in kk:
    int_area = i[0].intersection(i[1])
    overlap = int_area.area/i[0].area
    print(overlap)
    f200.write(str(overlap)+'\n')
f200.close
f500.close
    
