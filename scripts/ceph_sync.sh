#修改你的ceph地址,假设源地址id是/ceph/11027，目的地址/group/30116
ls -l /ceph/11027/|grep davidwwang| awk '{print $9}' > /data/dir.txt

while read line
do
    echo ${line} >> /data/unhandle.txt
done < /data/dir.txt

while read line
do
    echo "nohup rsync -ravz /ceph/11027/${line} /group/30116/ &"
    nohup rsync -ravz /ceph/11027/${line} /group/30116/ &
    sleep 1
done < /data/unhandle.txt

#ls -l /ceph/11027/ | awk '{print $9}' | xargs -I {} -P 60 -n 1 rsync -avh –progress /ceph/11027/{} /group/30116/
# 用rsync -avPh src_path des_path 做数据同步吧
