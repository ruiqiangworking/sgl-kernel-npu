# DeepEP Standalone Build

这是一个独立的 DeepEP 构建目录，包含了构建 DeepEP 内核和 Python 包所需的所有源码。

## 目录结构

```
deepep_standalone/
├── build.sh              # 独立构建脚本
├── README.md            # 本文档
├── csrc/                # C++ 源码
│   └── deepep/          # DeepEP 内核源码
│       ├── ops/         # ops 版本1
│       └── ops2/        # ops 版本2
└── python/              # Python 包源码
    └── deep_ep/         # DeepEP Python 包
```

## 构建前准备

确保已经正确安装并配置了 Ascend 工具链：

```bash
# 设置环境变量（可选，如果未设置，将使用默认路径）
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export SHMEM_HOME_PATH=/usr/local/Ascend/shmem/latest
```

## 构建命令

### 基本构建（使用 ops 版本1）

```bash
./build.sh
```

### 使用 ops2 版本构建

```bash
./build.sh -2
```

### 调试模式构建

```bash
./build.sh -d
```

### 指定 SOC 版本

```bash
./build.sh Ascend910_9382
```

### 组合选项

```bash
./build.sh -2 -d Ascend910_9382
```

## 构建选项

- `-2`: 使用 ops2 版本代替 ops 版本1
- `-d`: 启用调试模式
- `-h`: 显示帮助信息

## 构建输出

构建完成后，所有输出文件将生成在 `output/` 目录：

- `output/deep_ep*.whl` - DeepEP Python wheel 包
- `output/lib/` - 编译生成的共享库文件

## 安装

构建完成后，可以使用以下命令安装生成的包：

```bash
pip3 install output/deep_ep*.whl
```

或者强制重新安装：

```bash
pip3 install --force-reinstall output/deep_ep*.whl
```

## 构建步骤说明

1. **build_deepep_kernels**: 构建 DeepEP 内核算子
   - 编译 custom operators
   - 生成 .run 安装包
   - 将算子安装到 Python 包目录

2. **make_deepep_package**: 构建 DeepEP Python wheel 包
   - 清理旧的构建文件
   - 打包生成 wheel 文件
   - 将 wheel 文件移动到 output 目录

## 清理构建

如需清理构建文件：

```bash
rm -rf output/
rm -rf csrc/deepep/ops/build_out/
rm -rf csrc/deepep/ops2/build_out/
rm -rf python/deep_ep/build/
rm -rf python/deep_ep/dist/
rm -rf python/deep_ep/deep_ep.egg-info/
rm -rf python/deep_ep/deep_ep/vendors/
```

## 故障排除

### 找不到 Ascend 工具链

确保已安装 Ascend 工具链并正确设置了环境变量：

```bash
echo $ASCEND_HOME_PATH
```

### 构建失败

1. 检查 SOC_VERSION 是否正确
2. 确认所有必要的依赖已安装
3. 查看详细的错误日志

### 权限问题

确保 build.sh 有执行权限：

```bash
chmod +x build.sh
```

## 与主仓库的区别

这是一个精简的独立构建目录，仅包含：
- DeepEP 内核源码（csrc/deepep）
- DeepEP Python 包源码（python/deep_ep）

不包含其他模块如：
- sgl_kernel_npu
- torch_memory_saver
- 其他内核算子

如需完整构建，请使用主仓库根目录的 build.sh。
