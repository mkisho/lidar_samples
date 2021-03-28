#include "ros/ros.h"
#include "sensor_msgs/LaserScan.h"
#include "gazebo_msgs/SetLinkState.h"
#include "sensor_msgs/LaserScan.h"

#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream> 

#define NUM_SAMPLES 180
#define CENTRO_X 0
#define CENTRO_Y 0
#define CENTRO_Z 0.5

using namespace std;

sensor_msgs::LaserScan scan;

ofstream resultados("filename.txt");

void scanCallback(const sensor_msgs::LaserScan::ConstPtr& msg){
	scan.header = msg->header;
	scan.angle_min = msg->angle_min;
	scan.angle_max = msg->angle_max;
	scan.angle_increment =msg->angle_increment;
	scan.time_increment = msg->time_increment;
	scan.scan_time = msg->scan_time;
	scan.range_min = msg->range_min;
	scan.range_max = msg->range_max;
	scan.ranges = msg->ranges;
	scan.intensities = msg->intensities;
	
	for(int i=0; i < NUM_SAMPLES; i++){
		resultados << scan.ranges[i] << " ";
	}
	resultados << "true";
	resultados <<"\n";
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "lidar_samples");
  ros::NodeHandle n;
  ros::Subscriber sub = n.subscribe("scan", 1000, scanCallback);
  ros::ServiceClient client = n.serviceClient<gazebo_msgs::SetLinkState>("gazebo/set_link_state");
  gazebo_msgs::SetLinkState srv;
  srand(time(NULL));
  float x,y,z;
  ros::Rate loop_rate(3);
  for(int i=0; i<5000; i++){
	  x=((float)rand()/(float)(RAND_MAX)) * 2 + sin(i)*5 + CENTRO_X;
	  y=((float)rand()/(float)(RAND_MAX)) * 2 + cos(i)*5 + CENTRO_Y;
	  z=((float)rand()/(float)(RAND_MAX)) * 0.5 + CENTRO_Z;
	  
	  srv.request.link_state.link_name = "turtlebot3";
	  srv.request.link_state.pose.position.x = x;
	  srv.request.link_state.pose.position.y = y;
	  srv.request.link_state.pose.position.z = z;
	  srv.request.link_state.pose.orientation.x = 0;
	  srv.request.link_state.pose.orientation.y = 0;
	  srv.request.link_state.pose.orientation.z = 0;
	  srv.request.link_state.pose.orientation.w = 1;

	  srv.request.link_state.twist.linear.x = 0;
	  srv.request.link_state.twist.linear.y = 0;
	  srv.request.link_state.twist.linear.z = 0;

	  srv.request.link_state.twist.angular.x = 0;
	  srv.request.link_state.twist.angular.y = 0;
	  srv.request.link_state.twist.angular.z = 0;

	  srv.request.link_state.reference_frame = "world";

	  if (client.call(srv))
	  {
		
	  }
	  else
	  {
    		ROS_ERROR("Failed to call service lidar_samples");
    		return 1;
	  }
	  loop_rate.sleep();
  	  ros::spinOnce();
	  
	}
  resultados.close();
  return 0;
}
