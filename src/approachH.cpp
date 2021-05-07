#include "ros/ros.h"
#include "sensor_msgs/LaserScan.h"
#include "gazebo_msgs/SetLinkState.h"
#include "sensor_msgs/LaserScan.h"
#include "geometry_msgs/Quaternion.h"
#include <ros/package.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream> 

#define NUM_SAMPLES       2500
#define NUM_SAMPLES_CYCLE 720
#define CENTRO_X 0
#define CENTRO_Y 0
#define CENTRO_Z 5

#define PI 3.14159265359

using namespace std;

sensor_msgs::LaserScan scan;
int new_reading= 0;


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
	new_reading= 1;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "lidar_samples");
  if (argc != 3)
  {
    ROS_INFO("usage: lidar_samples_node File_name isTorre");
    return 1;
  }
  if(strcmp(argv[2], "true") && strcmp(argv[2], "false")){
    ROS_INFO("isTorre must be either true or false");
    return 1;
  }
  std::string path = ros::package::getPath("lidar_samples")+"/datasets/" + argv[1];
  cout << "Path do arquivo: " << path << "\n";
  ofstream resultados(path);

  ros::NodeHandle n;
  ros::Subscriber sub = n.subscribe("scan", 1, scanCallback);
  ros::ServiceClient client = n.serviceClient<gazebo_msgs::SetLinkState>("gazebo/set_link_state");
  gazebo_msgs::SetLinkState srv;
  srand(time(NULL));
  float x,y,z;
  float angle_sensor;
  int isValid=0;
  int isTorre=0;
  ros::Rate loop_rate(2);
  int i=0;
  float angle;
  tf2::Quaternion quat_tf;
  geometry_msgs::Quaternion q;
  x=-6;
  while(x<-5){
          x= -13 + i*0.04;
	  y= -4 + i*0.02;
	  z= 5;
	  angle_sensor=(((float)rand()/(float)(RAND_MAX)) * PI * 2) - PI + CENTRO_Z;
	  quat_tf.setRPY(0,0,PI); 
	  q = tf2::toMsg(quat_tf);
	  srv.request.link_state.link_name = "base_footprint";
	  srv.request.link_state.pose.position.x = x;
	  srv.request.link_state.pose.position.y = y;
	  srv.request.link_state.pose.position.z = z;
	  srv.request.link_state.pose.orientation = q;

	  srv.request.link_state.twist.linear.x = 0;
	  srv.request.link_state.twist.linear.y = 0;
	  srv.request.link_state.twist.linear.z = 0;

	  srv.request.link_state.twist.angular.x = 0;
	  srv.request.link_state.twist.angular.y = 0;
	  srv.request.link_state.twist.angular.z = 0;

	  srv.request.link_state.reference_frame = "world";
	  srv.response.success=false;
	  while (!client.call(srv))
	  {
		loop_rate.sleep();
    		ROS_WARN("Failed to call service lidar_samples");
		
	  }
	  loop_rate.sleep();
  	  ros::spinOnce(); 
  	  new_reading=false;
    	  ROS_INFO("Esperando msg\n");
	  while(!new_reading){
  	  	ros::spinOnce();

	  } 
          ROS_INFO("Msg received: %d", i);
          isValid=0;
	  for(int j=0; j < NUM_SAMPLES_CYCLE;j++){
	  	if(!isinf(scan.ranges[j])){
			isValid=1;
		}
	  }
          if(isValid){
	          ROS_INFO("Amostra Valida");

		  i++;
		  resultados << CENTRO_X - x << ", " << CENTRO_Y - y << ", ";
		  for(int j=0; j < NUM_SAMPLES_CYCLE; j++){
			if(isinf(scan.ranges[j]))
				resultados << "100" << ", ";
			else
				resultados << scan.ranges[j] << ", ";
		  }
		  resultados << argv[2];
   		  resultados <<"\n";
	 }
	}
  resultados.close();
  return 0;
}
