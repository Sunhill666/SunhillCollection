---
created: 2024-07-07
title: "Linux 下 CUDA & cuDNN 安装"
aliases:
  - "Linux 下 CUDA & cuDNN 安装"
tags:
  - linux
  - cuda
  - gpu
  - tutorial
---

> 本文以 Ubuntu 22.04/24.04、x86_64 架构为例。CUDA、驱动、编译器和 cuDNN 的兼容关系会随版本变化，安装前应始终查阅 NVIDIA 官方文档。

## 一、理解各组件的关系

- **NVIDIA Driver**：操作系统与 NVIDIA GPU 通信所需的驱动程序。
- **CUDA Toolkit**：包含 `nvcc`、开发头文件、CUDA 库和调试工具等。
- **cuDNN**：面向深度神经网络的 GPU 加速库，需要与 CUDA Toolkit、驱动和 GPU 架构兼容。

需要特别注意：

- `nvidia-smi` 显示的 **CUDA Version** 表示当前驱动能够支持的最高 CUDA 运行时版本，并不代表系统已经安装了对应版本的 CUDA Toolkit。
- `nvcc --version` 显示的是当前命令行使用的 CUDA Toolkit 版本。
- PyTorch、TensorFlow 等框架可能自带 CUDA 运行时，并不一定需要单独安装完整的 CUDA Toolkit；只有编译 CUDA 扩展或开发 CUDA 程序时通常才需要 Toolkit。

## 二、安装前检查

### 检查 NVIDIA GPU

```shell
lspci | grep -i nvidia
```

### 检查系统版本与架构

```shell
cat /etc/os-release
uname -m
uname -r
```

### 检查现有驱动与 CUDA

```shell
nvidia-smi
nvcc --version
```

如果命令不存在，不代表 GPU 有故障，只能说明对应的驱动或 Toolkit 尚未正确安装。

### 安装基础依赖

```shell
sudo apt update
sudo apt install -y build-essential linux-headers-$(uname -r) curl wget
```

## 三、安装 NVIDIA 驱动

NVIDIA 推荐优先使用发行版软件包管理器安装驱动，因为这种方式能与系统的包管理和内核升级流程集成。不要混用 APT 安装与 `.run` 安装，否则后续升级和卸载容易发生冲突。

### 推荐方式：使用 Ubuntu 驱动工具

查看推荐驱动：

```shell
ubuntu-drivers devices
```

自动安装推荐驱动：

```shell
sudo ubuntu-drivers autoinstall
sudo reboot
```

重启后验证：

```shell
nvidia-smi
```

如需严格控制驱动分支、安装开源内核模块或部署计算节点，请参考 [NVIDIA Driver Installation Guide for Ubuntu](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/ubuntu.html)。

## 四、处理 Nouveau 冲突

使用软件包管理器安装 NVIDIA 驱动时，安装程序通常会自动处理 Nouveau。只有驱动安装明确报告 Nouveau 冲突时，才建议手动禁用。

> 修改内核模块配置可能导致重启后图形界面不可用。操作前请确保可以通过 SSH 或虚拟终端登录系统。

创建配置文件：

```shell
sudo vim /etc/modprobe.d/blacklist-nouveau.conf
```

写入以下内容：

```text
blacklist nouveau
options nouveau modeset=0
```

重新生成 initramfs 并重启：

```shell
sudo update-initramfs -u
sudo reboot
```

验证 Nouveau 是否已禁用：

```shell
lsmod | grep nouveau
```

如果没有输出，表示 Nouveau 模块当前未加载。

若重启后无法进入图形界面，可以通过 SSH 登录，或按 `Ctrl + Alt + F3` 等组合键切换到虚拟终端。不同桌面环境使用的功能键可能不同，不应假定 `F7` 一定返回图形界面。

## 五、选择 GCC 版本

CUDA 对宿主编译器的支持范围取决于具体 CUDA 与 Linux 版本。不要仅因为版本号较新就强制把系统默认 GCC 切换为 GCC 12，更不要将不存在的“CUDA 14.1”作为判断依据。

首先检查当前编译器：

```shell
gcc --version
```

然后在 [CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) 中确认目标 CUDA 版本支持的 GCC 范围。如果确实需要其他版本，可以并行安装，例如：

```shell
sudo apt install gcc-12 g++-12
```

编译单个 CUDA 项目时，优先显式指定宿主编译器，避免修改整个系统的默认 GCC：

```shell
nvcc -ccbin /usr/bin/g++-12 example.cu -o example
```

## 六、安装 CUDA Toolkit

### 推荐方式：APT 软件包

1. 进入 [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)，选择所需版本。
2. 在安装选择器中选择对应的 Linux 发行版、版本、架构以及 `deb (network)`。
3. 严格执行页面生成的仓库配置命令。

![CUDA 下载页面的安装选项](attachments/Linux%20下%20CUDA%20&%20cuDNN%20安装/01-cuda-download-options.png)

_图 1：CUDA Toolkit 下载页面中的平台与安装方式选择。_

添加 NVIDIA CUDA 仓库后，更新 APT 缓存：

```shell
sudo apt update
```

安装仓库中的默认 CUDA Toolkit：

```shell
sudo apt install cuda-toolkit
```

如果需要固定版本，可安装带版本号的元包；具体包名以安装选择器和 APT 仓库为准，例如：

```shell
apt-cache search '^cuda-toolkit-[0-9]'
sudo apt install cuda-toolkit-<major>-<minor>
```

使用 `cuda-toolkit` 元包只安装 Toolkit。不要在已有可用驱动时盲目安装会同时拉取驱动的元包。

### 备选方式：Runfile

Runfile 适用于需要自定义安装目录或发行版软件包不可用的场景。常规 Ubuntu 环境更推荐使用 APT。

```shell
sudo sh cuda_<version>_linux.run
```

安装时注意：

- 如果 NVIDIA 驱动已经通过 APT 正常安装，应取消 Runfile 中的 Driver 选项，只安装 CUDA Toolkit。
- 不要混用 Runfile 与 APT 管理同一套驱动或 Toolkit。
- 使用 Runfile 安装驱动前，通常需要先禁用 Nouveau 并退出图形会话。

## 七、配置环境变量

若安装程序创建了 `/usr/local/cuda` 符号链接，可在 `~/.bashrc` 或 `~/.zshrc` 中加入：

```shell
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
```

使用 Runfile 安装时，通常还需要配置动态库路径：

```shell
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

使配置立即生效：

```shell
source ~/.bashrc
# 或
source ~/.zshrc
```

如果没有 `/usr/local/cuda` 链接，请将路径替换为实际安装目录，例如 `/usr/local/cuda-12.9`。

## 八、验证 CUDA 安装

```shell
nvidia-smi
nvcc --version
```

还可以检查 CUDA 安装目录：

```shell
ls -l /usr/local/cuda
```

如需进行完整功能测试，可编译运行 [NVIDIA CUDA Samples](https://github.com/NVIDIA/cuda-samples) 中的 `deviceQuery`。

## 九、安装 cuDNN

### 检查兼容性

安装前必须检查 [cuDNN Support Matrix](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/support-matrix.html)，确认以下组件相互兼容：

- cuDNN 版本；
- CUDA Toolkit 版本；
- NVIDIA Driver 版本；
- GPU 计算能力；
- Linux 发行版与架构。

### 使用 APT 安装 cuDNN 9

如果已经按照 CUDA 官方安装选择器添加 NVIDIA 仓库，可以直接更新索引：

```shell
sudo apt update
```

根据 CUDA 主版本选择对应的 cuDNN 包。当前官方文档提供的常用包包括：

```shell
# CUDA 12
sudo apt install -y cudnn9-cuda-12

# CUDA 13
sudo apt install -y cudnn9-cuda-13
```

> cuDNN 9 的不同 CUDA 主版本包不能同时安装。使用旧版 CUDA（例如 CUDA 11）时，应查阅与该 CUDA 版本对应的 cuDNN 历史文档和归档包，而不是直接套用当前命令。

安装后检查：

```shell
dpkg -l | grep cudnn
ldconfig -p | grep libcudnn
```

cuDNN 的完整安装方式与包名可参考 [Installing cuDNN Backend on Linux](https://docs.nvidia.com/deeplearning/cudnn/installation/latest/linux.html)。

## 十、常见问题

### `nvidia-smi` 正常，但 `nvcc` 不存在

说明 NVIDIA 驱动已经安装，但 CUDA Toolkit 未安装，或 Toolkit 的 `bin` 目录没有加入 `PATH`。

### `nvcc` 报告不支持当前 GCC

先确认目标 CUDA 版本支持的 GCC 范围，再安装兼容编译器，并通过 `nvcc -ccbin` 为项目单独指定。不要轻易使用 `--override` 绕过兼容性检查。

### 驱动安装后仍无法加载

检查以下项目：

```shell
lsmod | grep -E 'nvidia|nouveau'
dkms status
mokutil --sb-state
journalctl -k | grep -iE 'nvidia|nouveau'
```

常见原因包括 Nouveau 冲突、DKMS 构建失败、缺少当前内核头文件，以及 Secure Boot 阻止未签名的内核模块加载。

### 多个 CUDA 版本共存

多个 Toolkit 可以安装在 `/usr/local/cuda-X.Y` 下，通过调整 `PATH`、`LD_LIBRARY_PATH` 和 `/usr/local/cuda` 符号链接选择版本。切换前应确认目标项目及 cuDNN 与该版本兼容。

## 参考资料

- [CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)
- [NVIDIA Driver Installation Guide for Ubuntu](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/ubuntu.html)
- [Installing cuDNN Backend on Linux](https://docs.nvidia.com/deeplearning/cudnn/installation/latest/linux.html)
- [cuDNN Support Matrix](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/support-matrix.html)
