#!/bin/bash

 ps -ef | grep "forge_load_cpu" |  ps -ef | grep "forge_load_cpu"|awk '{print $2}'|xargs kill -9

