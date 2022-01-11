---
title: VScode Remote Setting
date: 2022-01-11 15:00:00
tags: [Tool]
categories: System
---

## VScode Remote

Visual Studio Code Remote 允许开发者将容器，远程计算机，或 Windows Subsystem for Linux (WSL) 作为完整的开发环境

![image](https://cdn.jsdelivr.net/gh/Trouble404/Image/20220111205110.png)

<!-- more -->

### General Install

安装Remote Development插件（在vscode下Ctrl/Command + Shift + X搜索下载安装)

* 配置config
![image](https://cdn.jsdelivr.net/gh/Trouble404/Image/20220111210051.PNG)

![image](https://cdn.jsdelivr.net/gh/Trouble404/Image/20220111210220.PNG)

![image](https://cdn.jsdelivr.net/gh/Trouble404/Image/20220111210333.PNG)

* 设置中勾选Show Log Terminal
![image](https://cdn.jsdelivr.net/gh/Trouble404/Image/20220111210427.PNG)

* 连接Remote进行开发

### Windows坑

- 本地.ssh权限问题
  - 找到.ssh文件夹。它通常位于C:\Users
  - 右键单击.ssh文件夹，然后单击“属性”。
  - 找到并点击“安全”标签。
  - 然后单击“高级”。 单击“禁用继承”，单击“确定”。 将出现警告弹出窗口。单击“从此对象中删除所有继承的权限”。
  - 接下来，单击“添加”以显示“选择用户或组”窗口。
  - 搜索用户名，添加用户权限

- Remote Extension 无法安装
  - ```settion.json``` 中使用git的ssh进行替换 ```"remote.SSH.path": "C:\\Program Files\\Git\\usr\\bin\\ssh.exe"```