import streamlit as st

st.title("这是一个标题")
st.header("这是一个较小的标题")
st.subheader("这是一个相对较小的标题")


st.markdown('''
# 静夜思
床前**明月**光，疑是地上霜。
举头望**明月**，低头思故乡。
''')

st.text('''
静夜思
床前明月光，疑是地上霜。
举头望明月，低头思故乡。
''')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 字符串
st.write("这是一段文本。")

# 数字
st.write(42)

# 列表
st.write([1, 2, 3])

# 字典
st.write({"key": "value"})

# 数据框（DataFrame）
df = pd.DataFrame({"Column 1": [1, 2, 3], "Column 2": ["A", "B", "C"]})
st.write(df)

# 多参数用法
st.write("这是一个字符串", 42, [1, 2, 3], {"key": "value"})

# 自定义渲染
fig, ax = plt.subplots()
x = np.linspace(0, 10, 100)
y = np.sin(x)
ax.plot(x, y)
st.write(fig)

data = {
    'latitude': [37.7749, 34.0522, 40.7128],
    'longitude': [-122.4194, -118.2437, -74.0060],
    'name': ['San Francisco', 'Los Angeles', 'New York']
}

st.map(data, zoom=4, use_container_width=True)
# streamlit run fdssd.py