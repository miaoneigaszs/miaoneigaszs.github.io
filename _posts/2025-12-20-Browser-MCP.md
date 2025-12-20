---
title: "Browser MCP"
categories:
  - MCP
tags:
  - MCP
  - Browser
toc: true
toc_sticky: true
toc_label: "目录"
toc_icon: "cog"
---
搭建了一个可以用自然语言来控制浏览器的Agent，其中使用了MCP工具`playwright`，使用`mcp_agent`来搭建client，使用的工具不再需要和客户端绑定，使用起来更加安全、灵活。{：.notice--primary}

这篇博客还将介绍有关MCP的基础知识{: .notice--info} 

## 什么是MCP
**MCP**（Model Context Protocol, 模型上下文协议），是Anthropic公司在2024年11月推出的开放标准协议，目的是规范LLM与外部工具、系统和数据源之间交互的方式。它提供了一种统一的接口，用于读取文件、执行函数和处理上下文提示。（<a href="https://en.wikipedia.org/wiki/Model_Context_Protocol" target="_blank">wikipedia</a>）目前，基本上所有的主流模型都支持MCP。

这张图示可以更清晰理解MCP的概念：
![MCP](/assets/images/What's-MCP.avif)

MCP