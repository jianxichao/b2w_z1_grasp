#!/usr/bin/env python3
"""Read-only SLAM/costmap snapshot with robot-centered comparison plot."""
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np


def main():
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from nav_msgs.msg import OccupancyGrid
    from geometry_msgs.msg import PolygonStamped
    from sensor_msgs.msg import LaserScan
    from tf2_ros import Buffer,TransformListener
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'output/map_diagnosis')
    parser.add_argument('--timeout',type=float,default=4.)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    rclpy.init();node=rclpy.create_node('navigation_map_readonly_snapshot')
    buffer=Buffer();listener=TransformListener(buffer,node);data={};received={};subs=[]
    topics=['/navigation_map','/global_costmap/costmap','/local_costmap/costmap']
    def receive(topic,msg):data[topic]=msg;received[topic]=time.monotonic()
    for topic in topics:
        subs.append(node.create_subscription(OccupancyGrid,topic,lambda m,t=topic:receive(t,m),qos_profile_sensor_data))
    subs.append(node.create_subscription(PolygonStamped,'/global_costmap/published_footprint',
        lambda m:receive('footprint',m),qos_profile_sensor_data))
    subs.append(node.create_subscription(LaserScan,'/scan',lambda m:receive('scan',m),qos_profile_sensor_data))
    try:
        deadline=time.monotonic()+args.timeout
        while time.monotonic()<deadline:rclpy.spin_once(node,timeout_sec=.05)
        if not all(t in data for t in topics):
            raise RuntimeError('Missing map topics: '+', '.join(t for t in topics if t not in data))
        tf=buffer.lookup_transform('map','base_footprint',rclpy.time.Time())
        p,q=tf.transform.translation,tf.transform.rotation
        yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
        report={'robot_map_xy_m':[p.x,p.y],'robot_yaw_rad':yaw,'topics':{}}
        arrays={}
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        from matplotlib.colors import ListedColormap
        fig,axes=plt.subplots(1,3,figsize=(16,5.5),layout='constrained')
        for index,(topic,ax) in enumerate(zip(topics,axes)):
            m=data[topic];ox=m.info.origin.position.x;oy=m.info.origin.position.y;res=m.info.resolution
            o=m.info.origin.orientation
            if m.header.frame_id!='map' or abs(o.x)+abs(o.y)+abs(o.z)>1e-6:
                raise RuntimeError('Snapshot plot requires axis-aligned grids in map frame')
            grid=np.asarray(m.data).reshape(m.info.height,m.info.width);arrays[f'grid_{index}']=grid
            cx=math.floor((p.x-ox)/res);cy=math.floor((p.y-oy)/res)
            cell=int(grid[cy,cx]) if 0<=cx<m.info.width and 0<=cy<m.info.height else None
            rows,cols=np.where(grid>=100);dist=np.hypot(ox+(cols+.5)*res-p.x,oy+(rows+.5)*res-p.y)
            nearest=None
            if len(dist):
                i=int(np.argmin(dist));nearest={'distance_m':float(dist[i]),
                    'map_xy_m':[float(ox+(cols[i]+.5)*res),float(oy+(rows[i]+.5)*res)]}
                ax.plot(*nearest['map_xy_m'],'mx',ms=10,mew=2)
            report['topics'][topic]={'robot_cell':cell,'nearest_occupied':nearest,
                'origin_xy_m':[ox,oy],'resolution_m':res,'shape':list(grid.shape),
                'received_age_sec':time.monotonic()-received[topic]}
            extent=[ox,ox+m.info.width*res,oy,oy+m.info.height*res]
            if index==0:
                cmap=ListedColormap(['.6','white','black'])
                shown=np.where(grid<0,0,np.where(grid>=65,2,1))
                ax.imshow(shown,origin='lower',extent=extent,cmap=cmap,vmin=0,vmax=2,interpolation='nearest')
            else:
                cmap=plt.get_cmap('turbo').copy();cmap.set_bad('.6')
                im=ax.imshow(np.ma.masked_less(grid,0),origin='lower',extent=extent,cmap=cmap,
                             vmin=0,vmax=100,interpolation='nearest')
                bar=fig.colorbar(im,ax=ax,shrink=.65,ticks=[0,50,99])
                bar.ax.set_yticklabels(['0','50','99–100'])
            foot=data.get('footprint')
            if foot is not None and foot.header.frame_id=='map':
                ax.add_patch(Polygon([(a.x,a.y) for a in foot.polygon.points],closed=True,
                                     fill=False,edgecolor='cyan',lw=2,label='Nav2 footprint'))
            ax.plot(p.x,p.y,'b+',ms=13,mew=2)
            ax.arrow(p.x,p.y,.35*math.cos(yaw),.35*math.sin(yaw),width=.025,color='blue')
            ax.set(xlim=(p.x-2,p.x+2),ylim=(p.y-2,p.y+2),xlabel='map X (m)',ylabel='map Y (m)',
                   title=f'{["SLAM map","Global costmap","Local costmap"][index]} | robot cell={cell}')
            ax.set_aspect('equal');ax.grid(alpha=.2)
        fig.suptitle('Blue: robot pose | Cyan: Nav2 footprint | Magenta: nearest occupied cell\n'
                     'SLAM: white=free, black=occupied, gray=unknown | Costmaps: 99=inscribed, 100=lethal')
        fig.savefig(args.output/'comparison.png',dpi=150);plt.close(fig)
        m=data['/navigation_map'];res=m.info.resolution;ox=m.info.origin.position.x;oy=m.info.origin.position.y
        grid=arrays['grid_0'];shown=np.where(grid<0,0,np.where(grid>=65,2,1))
        fig,ax=plt.subplots(figsize=(9,8),layout='constrained')
        ax.imshow(shown,origin='lower',extent=[ox,ox+m.info.width*res,oy,oy+m.info.height*res],
                  cmap=ListedColormap(['.6','white','black']),vmin=0,vmax=2,interpolation='nearest')
        ax.plot(p.x,p.y,'b+',ms=12,mew=2)
        ax.arrow(p.x,p.y,.4*math.cos(yaw),.4*math.sin(yaw),width=.03,color='blue')
        foot=data.get('footprint')
        if foot is not None and foot.header.frame_id=='map':
            ax.add_patch(Polygon([(a.x,a.y) for a in foot.polygon.points],closed=True,
                                 fill=False,edgecolor='cyan',lw=2))
        ax.set(title='SLAM map | white=free, black=occupied, gray=unknown',xlabel='map X (m)',ylabel='map Y (m)')
        ax.set_aspect('equal');ax.grid(alpha=.2)
        fig.savefig(args.output/'slam_map.png',dpi=150);plt.close(fig)
        np.savez_compressed(args.output/'grids.npz',**arrays)
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2));print('Saved:',args.output/'comparison.png')
    finally:
        node.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
