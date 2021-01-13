---
title: 使用Docker部署GitLab-CI-Runner
date: 2021-01-11 12:00:00
tags: [Linux]
categories: System
---

## GitLab-CI
![image](https://cdn.jsdelivr.net/gh/Trouble404/Image/blog20210111160221.png)
GitLab CI是为GitLab提供持续集成服务的一整套系统。在GitLab8.0以后的版本是默认集成了GitLab-CI并且默认启用的。
使用GitLab CI需要在仓库跟目录创建一个gitlab-ci.yml的文件，它用来指定持续集成需要运行的环境，以及要执行的脚本。还需要设置一个gitlab-runner，当有代码push变更的时候，gitlab-runner会自动开始pipeline，并在gitlab上显示持续集成的结果。

<!-- more -->
GitLab-Runner执行情况如下：
1. 本地代码改动
2. 变动代码推送到GitLab上
3. GitLab 将这个变动通知GitLab-CI
4. GitLab-CI找出这个工程相关联的gitlab-runner
5. gitlab-runner把代码更新到本地
6. 根据预设置的条件配置好环境
7. 根据预定义的脚本(一般是.gitlab-ci.yml)执行
8. 把执行结果通知给GitLab
9. GitLab显示最终执行的结果

gitlab-runner可以在不同的主机上部署，也可以在同一个主机上设置多个gitlab-runner ,还可以根据不同的环境设置不同的环境，比如我们需要区分研发环境，测试环境以及正式环境等。

## GitLab-Runner
Gitlab CI 对 Docker 的支持非常好，文档之类的东西非常全面，建议直接阅读官方文档即可：

[Run GitLab Runner in a container](https://docs.gitlab.com/runner/install/docker.html)
[Docker section of Registering Runners](https://docs.gitlab.com/runner/register/index.html#docker)

### 创建 Runner 容器
创建一个 gitlab-runner-docker 目录，然后新建一个 docker-compose.yml 文件，内容如下：
```sh
version: "3"
services:
  app:
    image: gitlab/gitlab-runner
    container_name: gitlab-runner-docker
    restart: always
    volumes:
      - ./config:/etc/gitlab-runner
      - /var/run/docker.sock:/var/run/docker.sock
```
在 gitlab-runner-docker 目录下执行 ```docker-compose up --build -d``` 命令，```docker ps -a``` 即可看见刚才创建的容器，同时目录下会生成一个 config 目录，用于存放 Runner 的配置文件

### 注册 Runner
执行命令 ```docker exec -it gitlab-runner-docker gitlab-runner register``` （或者进入容器内部执行 ```gitlab-runner register``` 也可以）

接下来会看到一系列的输入项，一步一步输入即可。

* 输入 GitLab 的地址。
如果是使用的官方的 GitLab，就输入 https://gitlab.com，自建的 GitLab 就输入自己的IP或域名即可。
```sh
Please enter the gitlab-ci coordinator URL (e.g. https://gitlab.com )
https://gitlab.com
```

*  输入 Token 来注册 Runner
在 GitLab 的仓库 Setting -> CI/CD 设置页面中，展开 Runners 部分，即可看到生成的 Token，复制粘贴即可。
![image](https://cdn.jsdelivr.net/gh/Trouble404/Image/blog20210111160945.png)
```sh
Please enter the gitlab-ci token for this runner
xxxToken
```

* 输入一段 Runner 的描述，之后可以在 GitLab 管理页面进行修改。
```sh
Please enter the gitlab-ci description for this runner
[hostame] my-runner
```

* 输入 Runner 关联的标签，之后可以在 GitLab 管理页面进行修改
```sh
Please enter the gitlab-ci tags for this runner (comma separated):
my-tag,another-tag
```

* Enter the Runner executor
GitLab Runner 内置了多种 executor，不同类型的 executor 的区别，可以参考文档：[Executors](https://docs.gitlab.com/runner/executors/README.html)。
这里我们填写 docker。
```sh
Please enter the executor: ssh, docker+machine, docker-ssh+machine, kubernetes, docker, parallels, virtualbox, docker-ssh, shell:
 docker
```

* If you chose Docker as your executor, you’ll be asked for the default image to be used for projects that do not define one in .gitlab-ci.yml
如果选择了 executor 为 docker，那么就需要选择一个默认的镜像。
    <br>

    官方文档的例子中，默认镜像使用的是 alpine:latest，是个精简版的 linux 镜像，也就意味着啥工具都没有，每次执行构建任务前先装 git 等一系列工具显然不合理。

    <br>

    推荐使用 tico/docker 作为默认镜像，这个镜像在官方 docker 镜像的基础上，加入了 curl、php、git 等等一系列常用的工具。

    <br>

    也可以使用本地的镜像，这样免去每次配置的时间(提前准备好一个配置好测试环境的镜像)

<br>

### 配置 Runner
进入 config 目录，会发现一个 config.toml 文件，里面是 gitlab-runner 相关的配置信息。
```sh
concurrent = 1
check_interval = 0

[session_server]
  session_timeout = 1800

[[runners]]
  name = "home-runner-docker"
  url = "https://gitlab.com"
  token = "xxxxxxxxxxxxxxx"
  executor = "docker"
  [runners.docker]
    tls_verify = false
    image = "tico/docker"
    privileged = false
    disable_entrypoint_overwrite = false
    oom_kill_disable = false
    disable_cache = false
    volumes = ["/cache"]
    shm_size = 0
  [runners.cache]
    [runners.cache.s3]
    [runners.cache.gcs]
```

当执行构建任务时如果出现以下报错，请将上面的配置文件中的 privileged 的值改为 true。
![image](https://cdn.jsdelivr.net/gh/Trouble404/Image/blog20210111161741.png)

需要使用**本地镜像**的话, 添加pull_policy到配置文件中
```sh
concurrent = 1
check_interval = 0

[session_server]
  session_timeout = 1800

[[runners]]
  name = "home-runner-docker"
  url = "https://gitlab.com"
  token = "xxxxxxxxxxxxxxx"
  executor = "docker"
  [runners.docker]
    tls_verify = false
    image = "tico/docker"
    privileged = false
    disable_entrypoint_overwrite = false
    oom_kill_disable = false
    disable_cache = false
    volumes = ["/cache"]
    **pull_policy = "if-not-present"**
    shm_size = 0
  [runners.cache]
    [runners.cache.s3]
    [runners.cache.gcs]
```

### 使用
以博客为例，首先是一个专门用来执行构建任务 Runner 容器。

当接收到博客的构建任务时，创建一个基于 tico/docker 镜像的容器，然后再在这个容器中执行构建脚本。

而构建脚本中，又调用 docker 命令创建了一个 nodejs 容器来进行打包编译，然后再将 build 之后生成的静态文件移入一个 nginx 镜像，作为最终部署使用的镜像并上传到阿里云容器服务。

接着 GitLab CI 会将部署任务发送至另一个专门用来执行部署任务的 Runner 容器，这个 Runner 容器会 ssh 登录上目标服务器，拉取最新的镜像并运行。

.gitlab-ci.yml 例子:

```sh
stages:
  - build
  - test
  - release

# 构建
build:
  stage: build
  tags:
    - xs1
    - docker
  image: tico/docker:latest
  services:
    - docker:dind
  variables:
    DOCKER_HOST: tcp://docker:2375/
    DOCKER_DRIVER: overlay2
    DOCKER_REPO: $docker_repo
  before_script:
    - echo 'before_script'
    - docker login --username=$docker_user -p $docker_pwd $docker_host
  script:
    - sh ./ci/build/build.sh
  only:
    - master

# 发布
release:
  stage: release
  tags:
    - xs1
    - ssh
  variables:
    DOCKER_REPO: $docker_repo
    PROJECT_NAME: $project_name
    SERVER: $deploy_server
    PORT: $port
  script:
    - sh ./ci/release/release.sh
  environment:
    name: production
  only:
    - master
```

结果:
![image](https://cdn.jsdelivr.net/gh/Trouble404/Image/blog20210111162334.png)