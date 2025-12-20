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
我做了一个能“听懂人话”的浏览器 Agent！它利用 MCP 协议连接了 LLM 和 Playwright。最大的收获是理解了 MCP 的解耦思想——工具运行在独立的服务端，不再硬编码在客户端里，这让整个系统变得更加安全和灵活。
{: .notice--primary}

这篇博客还将介绍有关MCP的基础知识。
{: .notice--warning}

## 什么是MCP
**MCP**（Model Context Protocol, 模型上下文协议），是Anthropic公司在2024年11月推出的开放标准协议，目的是规范LLM与外部工具、系统和数据源之间交互的方式。它提供了一种统一的接口，用于读取文件、执行函数和处理上下文提示。（<a href="https://en.wikipedia.org/wiki/Model_Context_Protocol" target="_blank">wikipedia</a>）目前，基本上所有的主流模型都支持MCP。

这张图示可以更清晰理解MCP的概念：

[![MCP](/assets/images/What's-MCP.avif)](/assets/images/What's-MCP.avif)
*图 1：MCP 架构示意图*
{: .text-center .caption}

这个动图显示了MCP的优势：
[![MCP-gif](/assets/images/What%20is%20MCP.gif)](/assets/images/What%20is%20MCP.gif)
*图 2：MCP 优势*
{:.text-center .caption}

## MCP架构和组件

MCP架构由三部分组成：
1. MCP宿主（MCP Host）
   MCP宿主代表提供交互环境的AI应用程序，宿主充当运行MCP客户端的环境，并提供用户与AI进行交互的界面。
2. MCP客户端（MCP Client）
   MCP客户端在宿主机上运行，负责协调与MCP服务器的通信。其职责包括：
   - 与MCP服务器建立连接
   - 发送请求
   - 接收响应
3. MCP服务器（MCP Server）
   主要对外提供三种能力：
   - 工具（Tools）：使大模型可以执行操作（如搜索数据库、调用API等）
   - 资源（Resources）：可供大模型访问的数据和内容（如文档、结构化数据等）
   - 提示词（Prompts）：用于生成特定类型的工作流或可重用模板

## MCP的工作原理

