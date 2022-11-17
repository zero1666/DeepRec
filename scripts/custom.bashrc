#!/bin/bash

# 默认个人代码空间
export ENV_WORKSPACE_USER=/workspace/user_code/davidwwang/workspace/
export PS1="\[\e[0;33m\][\u@$(/sbin/ifconfig eth1|grep inet|awk '{print $2}'|cut -d: -f2) \[\e[0;35m\]\w]\$\[\e[0m\]"

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/data/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/data/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/data/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/data/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

