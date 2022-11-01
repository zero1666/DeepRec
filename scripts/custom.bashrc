#!/bin/bash

# 默认个人代码空间
export ENV_WORKSPACE_USER=/workspace/user_code/davidwwang/workspace/
export PS1="\[\e[0;33m\][\u@$(/sbin/ifconfig eth1|grep inet|awk '{print $2}'|cut -d: -f2) \[\e[0;35m\]\w]\$\[\e[0m\]"

