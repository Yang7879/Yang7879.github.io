# -*- coding: utf-8 -*-
import os
import numpy as np
import open3d as o3d


def vis_cloud(tmp,label=None):
    # 给多个矩阵赋予不同的颜色显示，传入数据是一个列表，列表的每一项包括了一个points_clouds变量，每一个变量都是n*3的np矩阵
    # 第二个参数 是label ，label 是 n*1的一个list。暂时假设label不超过6，如果有更多种类再添加color_set
    # 最多设置的颜色类型是color_set，暂时储存了6种颜色
    color_set=np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1],[1,0,1]])
    pt = []
    for i,t in enumerate(tmp):
        pt1=o3d.geometry.PointCloud()
        pt1.points=o3d.utility.Vector3dVector(t.reshape(-1,3))
        if(label is None):
            pt1.paint_uniform_color(color_set[i])
        else:
            pt1.paint_uniform_color(color_set[label[i]])
        pt.append(pt1)
    o3d.visualization.draw_geometries(pt,window_name='cloud[0] and corr',width=800,height=600)

# example
# points2 = np.random.rand(500,3)
# points3 = np.random.rand(500,3)
# points = []
# label = [3,5]
# points.append(points2)
# points.append(points3)
# vis_cloud(points,label)
