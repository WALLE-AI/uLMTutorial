# uLMTutorial
从0到1动手学习大模型技术

## :telescope:目录

### :computer:入门科普课程

|                           科普课程                           | 课程描述内容                                                 | 推荐值                                                   |
| :----------------------------------------------------------: | ------------------------------------------------------------ | -------------------------------------------------------- |
| [Recent Advances on Foundation Models](https://cs.uwaterloo.ca/~wenhuche/teaching/cs886/) | 滑铁卢大学Wenhu Chen老师的课程“Recent Advances on Foundation Models”在滑铁卢大学是公开的。课程中，覆盖了许多有趣的话题，包括**Transformers、LLM、预训练、量化、稀疏注意力、指令调整、RLHF、提示、视觉Transformers、扩散模型、多模态模型、代理、RAG**等。https://cs.uwaterloo.ca/~wenhuche/teaching/cs886/ | :yellow_heart::yellow_heart::yellow_heart:               |
| [stas00](https://github.com/stas00)/[ml-engineering](https://github.com/stas00/ml-engineering) | 《Machine Learning Engineering Open Book》的开源书籍，该书籍汇集了多种方法论，旨在帮助训练大型语言模型和多模态模型。文章指出，这本书适合用于大型语言模型（LLM）和超大型语言模型（VLM）的培训工程师和操作员，其中包含了许多脚本和复制粘贴命令，以便快速满足需求。文章还提到，这本书是一个持续的脑洞集合，作者在训练开源BLOOM-176B模型和IDEFICS-80B多模态模型时获得了许多专业知识。目前，作者正在Contextual.AI开发/训练开源检索增强模型。文章最后提到了该书籍的目录结构，包括见解、关键硬件组件、性能、操作、开发、杂项等部分。 | :yellow_heart::yellow_heart:                             |
| [langchain/cookbook](https://github.com/langchain-ai/langchain/tree/masterhttps://github.com/langchain-ai/langchain/tree/master/cookbook) | LangChain提供了一个名为cookbook的代码示例集合，可用于构建使用LangChain的应用程序。这个cookbook强调更多应用和端到端示例，而不是主要文档中包含的示例。其中包括一个构建聊天应用程序的示例，该应用程序可以使用开源的llm（llama2）与SQL数据库进行交互，具体示例展示了包含名单的SQLite数据库 | :yellow_heart::yellow_heart::yellow_heart:               |
| [rasbt](https://github.com/rasbt)/[LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) | 如何从零开始构建一个类似于ChatGPT的大型语言模型（LLM）。通过逐步的指导和清晰的文本、图表和例子，引导读者创建自己的LLM。文章还提到了训练和发展自己用于教育目的的小型但功能齐全的模型的方法，这与创建大型基础模型（如ChatGPT）的方法类似。文章还提供了章节标题和主要代码的快速访问链接 | :yellow_heart::yellow_heart::yellow_heart::yellow_heart: |
| [2024年构建大型语言模型的小指南](https://docs.google.com/presentation/d/1IkzESdOwdmwvPxIELYJi8--K3EZ98_cL6c5ZcLKSyVg/mobilepresent?pli=1&slide=id.g2c144c77cfe_0_352) | **Hugging Face创始人Thomas Wolf讲解**如何构建大型语言模型的文章。该文章可能提供了关于大型语言模型的定义、预训练过程、技术挑战和伦理关注点等内容。它还可能介绍了Meta AI在2024年推出的大型语言模型，并突出了其性能优势。文章可能还包括关于大型语言模型在自然语言处理和人工智能领域的应用以及构建这类模型所需的专业技能和知识然而，由于上下文提供的信息有限，无法对文章的具体内容进行进一步详细解释 | :yellow_heart::yellow_heart::yellow_heart:               |
| **[Hung-yi Lee GPT科普视频](https://www.youtube.com/@HungyiLeeNTU)**（油管） | **通俗易懂讲解什么chatgpt？什么GenAI,课程生动有趣**          | :yellow_heart::yellow_heart::yellow_heart:               |
| [datawhalechina](https://github.com/datawhalechina)/ [so-large-lm](https://github.com/datawhalechina/so-large-lm) | datawhale构建项目旨在作为一个大规模预训练语言模型的教程，从数据准备、模型构建、训练策略到模型评估与改进，以及模型在安全、隐私、环境和法律道德方面的方面来提供开源知识。项目将以[斯坦福大学大规模语言模型课程](https://stanford-cs324.github.io/winter2022/)和[李宏毅生成式AI课程](https://speech.ee.ntu.edu.tw/~hylee/genai/2024-spring.php)为基础，结合来自开源贡献者的补充和完善，以及对前沿大模型知识的及时更新，为读者提供较为全面而深入的理论知识和实践方法。通过对模型构建、训练、评估与改进等方面的系统性讲解，以及代码的实战，我们希望建立一个具有广泛参考价值的项目 | :yellow_heart::yellow_heart::yellow_heart:               |
| [CS224N: Natural Language Processing with Deep Learning](https://web.stanford.edu/class/cs224n/) | **从传统的NLP到大模型技术进行全面的讲解，其中包含各种小项目实践操作** | :yellow_heart::yellow_heart::yellow_heart::yellow_heart: |
| **[ Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g)** | **无代码的科普，讲述大模型如何通过算力压缩得到，**[ppt地址](https://drive.google.com/file/d/1pxx_ZI7O-Nwl7ZLNk5hI3WzAsTLwvNU7/view) | :yellow_heart::yellow_heart::yellow_heart:               |
| [mlabonne](https://github.com/mlabonne)/**[llm-course](https://github.com/mlabonne/llm-course)** | 该文章主要介绍了如何通过一个课程进入大型语言模型（LLMs）的学习，该课程分为三个部分：LLM基础、LLM科学家和LLM工程师。课程提供了相关的笔记本和文章，包括用于评估LLMs、合并模型、量化LLMs和其他相关主题的工具和资源，分别提供基础课程以及LLM科学家和工程师技术路线，课程具备了基础知识、理论阐述和项目实践操纵 | :yellow_heart::yellow_heart::yellow_heart:               |
| [microsoft](https://github.com/microsoft)/**[generative-ai-for-beginners](https://github.com/microsoft/generative-ai-for-beginners)** | 主要介绍了一个由微软云倡导者提供的12节课的课程，旨在帮助初学者学习生成式AI应用程序的开发。课程涵盖了生成式AI原理和应用程序开发的关键方面，通过学习，学生可以构建自己的生成式AI初创公司，以了解启动创意所需的条件。文章还提到了如何开始学习、与其他学习者交流和支持、进一步学习资源以及如何为课程做出贡献 | :yellow_heart::yellow_heart::yellow_heart::yellow_heart: |
| [复旦大学-《大规模语言模型：理论到实践》](https://intro-llm.github.io/) | **基本阐述大模型发展历史、技术细节和评估方式** [复旦大学张奇教授相关资料](https://zhuanlan.zhihu.com/p/670742372) | :yellow_heart::yellow_heart::yellow_heart:               |
|        [easywithai/ai-courses](easywithai/ai-courses)        | GenAI各种课程，从普通入门到进阶，基本都是付费/免费的课程培训，其他包括吴恩达 deeplearning-ai | :yellow_heart::yellow_heart::yellow_heart::yellow_heart: |

### :bomb:AI神器推荐---提高办公效率、上班摸鱼好利器

#### Prompt大魔法案例与技巧

| 文本生成类（OpenAI GPTs相关免费应用）                        | 描述                                                         | 推荐值                                                   |
| ------------------------------------------------------------ | ------------------------------------------------------------ | -------------------------------------------------------- |
| [promptbase](https://promptbase.com/)                        | Prompt商场，可以发布自己prompt进行售卖，也可以购买商城中prompt案例 | :yellow_heart::yellow_heart:                             |
| [Awesome-GPTs-Big-List](https://github.com/friuns2/Awesome-GPTs-Big-List) | 参考Openai GPTs 的Prompt写法                                 | :yellow_heart::yellow_heart::yellow_heart:               |
| [awesome-prompts](https://github.com/ai-boost/awesome-prompts) | Open GPTs Prompt聚集地，学习大量场景的prompt写法             | :yellow_heart::yellow_heart::yellow_heart::yellow_heart: |
| **[BlackFriday-GPTs-Prompts](https://github.com/friuns2/BlackFriday-GPTs-Prompts)** | 参考Openai GPTs 的Prompt写法                                 | :yellow_heart::yellow_heart::yellow_heart:               |
| **[Leaked-GPTs](https://github.com/friuns2/Leaked-GPTs)**    | 参考Openai GPTs 的Prompt写法                                 | :yellow_heart::yellow_heart::yellow_heart:               |

#### AI办公/摸鱼神器推荐

| 智能助手系列                                                 | 描述                                                         | 推荐值                                                   |
| ------------------------------------------------------------ | ------------------------------------------------------------ | -------------------------------------------------------- |
| [字节Coze](https://www.coze.cn/)                             | 扣子为你提供了一站式 AI 开发平台,无需编程，你的创新理念都能迅速化身为下一代的 AI 应用，可以快速无代码方式搭建自己GPT应用，缺点是只能使用字节本身的模型，比如云雀、豆包 | :yellow_heart::yellow_heart::yellow_heart::yellow_heart: |
| [POE](https://poe.com/)                                      | 提供主流的模型能力，比如Openai GPT4 Claude3 Midjourney，以及创作各个类型的助手，大部分是免费的，**需要梯子** | :yellow_heart::yellow_heart::yellow_heart:               |
| [Monica](https://monica.im/home)                             | 提供了GPT4服务，具备聊天、文档阅读、AI search、写作、翻译与艺术创作等较大的AI功能，支持PC端、浏览器端、移动端，缺点就是价格太贵，**需要梯子** | :yellow_heart::yellow_heart::yellow_heart::yellow_heart: |
| [百度超级助手](https://cloud.baidu.com/product/infoflow.html) | 超级助理是基于文心大模型的智能助手，含Copilot应用端和Agent开发平台两部分。Copilot以浏览器插件提供翻译、创作、总结、问答等知识服务，也可调用Agent开发平台插件自定义任务。Agent平台则支持插件注册与编排，快速打造场景化应用 | :yellow_heart::yellow_heart::yellow_heart:               |
| [智普AI 清言](https://chatglm.cn/)                           | 能够使用智普AI最新的GLM4模型，零代码方式构建智能体服务，具备生图、数据分析、文档分析、联网等功能 | :yellow_heart::yellow_heart::yellow_heart::yellow_heart: |
| [LOBE社区](https://github.com/lobehub/lobe-chat)             | LLMs/AI 聊天框架。它支持多个 AI 提供商（如 OpenAI、Claude 3、Gemini、Perplexity、Bedrock、Azure、Mistral、Ollama），以及多模态（视觉/TTS）和插件系统。它支持一键免费部署私有的 ChatGPT 聊天应用程序。该项目提供了详细的设置开发指南和资源文档，以及插件和主题定制的能力 **开源社区** | :yellow_heart::yellow_heart::yellow_heart:               |
| [Dify](https://github.com/langgenius/dify)                   | langgenius/dify是一个名为Dify的LLM应用开发平台，它结合了Backend-as-a-Service和LLMOps的概念，旨在帮助开发者快速构建高质量的生成式AI应用Dify提供了一个可视化编排工具，使应用的开发、运维和数据集管理更加简单它还集成了一个内置的RAG引擎，支持全文搜索和向量数据库嵌入的功能，并允许直接上传各种文本格式的文件Dify支持多种类型的应用，包括开箱即用的Web站点、表单模式和聊天对话模式的应用 **开源社区** | :yellow_heart::yellow_heart::yellow_heart:               |
| [easywithai](https://easywithai.com/)                        | chatbot 、生图 、视频生成、AI tools聚集地，各类AI产品导航栏与入口 | :yellow_heart::yellow_heart::yellow_heart::yellow_heart: |

| AI Search/Chat                              | 描述                                                         | 推荐值                                                   |
| ------------------------------------------- | ------------------------------------------------------------ | -------------------------------------------------------- |
| [天工AI](https://search.tiangong.cn/)       | copilot方式会对搜索**query进行分析（明显的子问题分析与扩展）**和关键字抽取，同时对搜索内容进行过滤与分析，可实现多轮search的方式接着询问 | :yellow_heart::yellow_heart::yellow_heart:               |
| [Kimi(月之暗面)](https://kimi.moonshot.cn/) | 暂时没有提供联网搜索的内容，支持本地文件上传，提供chatpdf的能力 | :yellow_heart::yellow_heart:                             |
| https://gptcall.net/                        | 通过代理方式接入大部分Openai GPTs应用，每天限时免费          | :yellow_heart::yellow_heart::yellow_heart::yellow_heart: |
| [秘塔AI](https://metaso.cn/)                | query分析，其实做子问题生成与问题扩展 调用搜索API接口或者其他搜索技术，搜索30条相关信息 对search result结果进行重排序和url内容总结，整理出搜索上下文 Prompt设计应该采用引用策略 延申功能做很多，比如大纲生成（包括脑图生成）、实体解读概述（主要是人名、机构等）、事件生成以及延申阅读 | :yellow_heart::yellow_heart::yellow_heart::yellow_heart: |
| [You](https://you.com/)                     | you.com的产品相对之前有较大改动，之前有chat、search 产品形态较为复杂，改版后，**直接通过chat方式完成多轮AI search 问题，这个方式跟openai web search plugin的方式类似**，**需要梯子** | :yellow_heart::yellow_heart::yellow_heart:               |

| 生图系列                                  | 描述                                                         | 推荐值                                                   |
| ----------------------------------------- | ------------------------------------------------------------ | -------------------------------------------------------- |
| [Midjourney](https://www.midjourney.com/) | 能够生成高质量图片，具有较强的仿真能力，月租价格较贵 **需要梯子** | :yellow_heart::yellow_heart::yellow_heart::yellow_heart: |
| [stability.ai](https://stability.ai/)     | 开源SDXL\SDXL Tubo等模型，也提供官方PROB版本模型，缺点是价格较贵 **需要梯子** | :yellow_heart::yellow_heart::yellow_heart:               |
| [DALLE-3](https://openai.com/dall-e-3)    | OpenAI 提供的生图模型服务，只能在ChatGPT或者GPTs的产品中使用 **需要梯子** | :yellow_heart::yellow_heart::yellow_heart:               |

| 视频生成系列                                      | 描述                                                         | 推荐值                                                   |
| ------------------------------------------------- | ------------------------------------------------------------ | -------------------------------------------------------- |
| [Runway](https://runwayml.com/)                   | Runway拥有30多种工具，可以轻松进行创意构思、生成和编辑内容，带来前所未有的体验。你可以免费试用它，无论你想创作什么，Runway都能满足你的需求 | :yellow_heart::yellow_heart::yellow_heart::yellow_heart: |
| [PIKA](https://pika.art/home)                     | Pika是一家重新设计视频制作和编辑体验的AI视频平台。他们最近推出了Pika 1.0版本，这是一次重大的产品升级。Pika 1.0拥有强大的AI模型，可以将照片、绘画和视频转化为沉浸式、动态的场景，并进行编辑和修改 | :yellow_heart::yellow_heart::yellow_heart:               |
| Sora                                              |                                                              |                                                          |
| [stable-video](https://stability.ai/stable-video) | Stable Video Diffusion，这是一个基于图像模型的生成式视频模型10。Stability AI的目标是提供一系列AI模型，涵盖图像、语言、音频、3D和代码等不同领域，以增强人类智能2。他们的产品适用于多个领域，如媒体、娱乐、教育和营销，并且可以将文本和图像转化为生动的场景，创作出生动的电影作品 | :yellow_heart::yellow_heart::yellow_heart:               |

| 文档问答/写作系列                    |      |      |
| ------------------------------------ | ---- | ---- |
| [askyoupdf](https://askyourpdf.com/) |      |      |
| [chatpdf](https://www.chatpdf.com/)  |      |      |
| [秘塔写作猫](https://xiezuocat.com/) |      |      |

###  :mag_right:数据收集与分析

### :chart_with_upwards_trend:应用技术

### :ledger:模型基础技术

### :dart:基础平台技术能力
