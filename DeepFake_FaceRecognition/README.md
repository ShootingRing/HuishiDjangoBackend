# 运行全过程

## 直接运行DETECT.py文件

### 输入用户名

在results文件夹下可以找到三个存储检测结果txt,以及一个人对应的三段视频

![image-20230406143710943](C:\Users\Fairy\AppData\Roaming\Typora\typora-user-images\image-20230406143710943.png)

如上图



### split为人像分割

在本文件中是视频调入分割

如需完成实时分割

可修改Spilt.eval.detect函数中args.video

```python
parse_args()
args.video=input
```



### 活体检测录制时间修改

```python
if (current_time - start_time) >=0.5 :
    print("视频录制完毕,时间为0.5秒")
    break
```