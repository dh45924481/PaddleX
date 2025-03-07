依赖库:
pyquaternion
fastapi
python-multipart

报错:
ImportError: /root/miniconda3/envs/paddlex/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by /root/miniconda3/envs/paddlex/lib/python3.10/site-packages/paddle/base/libpaddle.so)   代码在ubuntu下运行报错

解决:
1: conda install -c conda-forge gcc gxx
2: conda install -c conda-forge libstdc++
3: sudo apt-get update
sudo apt-get install libstdc++6

-------------------------------------------------------