鉴于LICENSE限制不能直接放出处理后的文件，请按照如下步骤预处理MANO模型文件

1. 对于原始的MANO_LEFT/RIGHT.pkl文件，请将其中的所有成员转换为np.array类型避免出现错误
2. 请分别从MANO_LEFT/RIGHT.pkl文件中提取hand_components成员（形状为45*45）存储为mano_lr_pca.npz

```python
# mano_lr_pca.npz
{
    "left": np.array,
    "right": np.array
}
```