---
title: PickGo 搭建图床
date: 2020-10-19 19:30:00
tags: [Deep Learning]
categories: System
---

![image](https://cdn.jsdelivr.net/gh/Trouble404/Image/blog20201019192744.jpg)

PicGo 算得上一款比较优秀的图床工具。它是一款用 Electron-vue 开发的软件，可以支持微博，七牛云，腾讯云COS，又拍云，GitHub，阿里云OSS，SM.MS，imgur 等8种常用图床，功能强大，简单易用

<!-- more -->

## Install

[软件发布地址](https://github.com/Molunerfinn/PicGo/releases)

* Mac 选择 dmg 下载
* Windows 选择 exe 下载
* Linux 选择 Appimage 下载

### Ubantu
PickGo 2.2.2

```shell
chmod a+x PicGo-2.2.2.AppImage
./PicGo-2.2.2.AppImage
```

## GitHub图床
首先登录GitHub，新建一个仓库或者也可以使用一个已有仓库(使用公有仓库)。
![image](https://cdn.jsdelivr.net/gh/Trouble404/Image/blog20201019193623.png)

创建好后，需要在 GitHub 上生成一个 token 以便 PicGo 来操作我们的仓库，来到个人中心，选择 Developer settings 就能看到 Personal access tokens，我们在这里创建需要的 token。点击 Generate new token 创建一个新 token，选择 repo，同时它会把包含其中的都会勾选上，我们勾选这些就可以了。然后拉到最下方点击绿色按钮，Generate token 即可。之后就会生成一个 token ，记得复制保存到其他地方，这个 token 只显示一次！！

![image](https://cdn.jsdelivr.net/gh/Trouble404/Image/blog20201019193838.png)

打开PicGo并且设置github图床
![image](https://cdn.jsdelivr.net/gh/Trouble404/Image/blog20201019194001.png)

之后就可以上传图片到我们设定的github图床了
![image](https://cdn.jsdelivr.net/gh/Trouble404/Image/blog20201019194150.png)

## Blog快速加载图片
JSDelivr 是一个在中国访问速度极快的公有CDN, 使用也特别方便。在需要引用的仓库图片图片地址前加入 **https://cdn.jsdelivr.net/gh/** 即可。

```shell
![image](https://cdn.jsdelivr.net/gh/Trouble404/Image/blog20201019194150.png)
```
