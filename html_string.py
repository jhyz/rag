main_html = """<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>本地RAG解决方案</title>
    <link rel="stylesheet" href="https://fonts.googleapis.com/icon?family=Material+Icons">
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #e3f2fd; /* 浅蓝色背景 */
            margin: 0;
            padding: 0;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        header {
            background-color: #1976d2; /* 蓝色背景 */
            color: black;
            width: 100%;
            padding: 1.5em;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        main {
            margin: 2em;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            background-color: white;
            border-radius: 8px;
            overflow: hidden;
            width: 90%;
            max-width: 800px;
            padding: 2em;
        }
        h1 {
            color: #1976d2; /* 蓝色文字 */
        }
        p {
            color: #1976d2; /* 蓝色文字 */
            font-size: 1.1em;
        }
        ul {
            list-style-type: none;
            padding: 0;
        }
        ul li {
            background-color: #1976d2; /* 蓝色背景 */
            margin: 0.5em 0;
            padding: 1em;
            border-radius: 4px;
            transition: background-color 0.3s;
        }
        ul li a {
            color: white;
            text-decoration: none;
            display: flex;
            align-items: center;
        }
        ul li:hover {
            background-color: #1565c0; /* 加深蓝色 */
        }
        .material-icons {
            margin-right: 0.5em;
        }
    </style>
</head>
<body>
    <header>
        <h1>本地RAG解决方案</h1>
    </header>
    <main>
        <p>————————毕业设计</p>
        <ul>
            <li><a href="/upload_data"><span class="material-icons"></span> 1. 上传数据</a></li>
            <li><a href="/create_knowledge_base"><span class="material-icons"></span> 2. 创建知识库</a></li>
            <li><a href="/chat"><span class="material-icons"></span> 3. 基于LlamaIndex的RAG问答</a></li>
            <li><a href="http://127.0.0.1/chat/share?shared_id=4ea39760cf0511efb57f0242ac120006&from=chat&auth=E4ZDUyMWVlM2E5YTExZjBiODMzMDI0Mm"><span class="material-icons"></span> 4. 基于RAGFlow的RAG问答</a></li>
        </ul>
    </main>
</body>
</html>"""

plain_html = """<!DOCTYPE html>
<html lang="zh">
    <head>
        <title>RAG问答</title>
        <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
        <style>
        .links-container {
            display: flex;
            justify-content: center; /* 在容器中居中分布子元素 */
            list-style-type: none; /* 去掉ul默认的列表样式 */
            padding: 0; /* 去掉ul默认的内边距 */
            margin: 0; /* 去掉ul默认的外边距 */
        }
        .links-container li {
            margin: 0 5px; /* 每个li元素的左右留出一些空间 */
            padding: 10px 15px; /* 添加内边距 */
            border: 1px solid #1976d2; /* 蓝色边框 */
            border-radius: 5px; /* 添加圆角 */
            background-color: #ffffff; /* 白色背景 */
            transition: background-color 0.3s; /* 背景颜色变化的过渡效果 */
            display: flex; /* 使用flex布局 */
            align-items: center; /* 垂直居中对齐 */
            height: 50px; /* 设置固定高度，确保一致 */
        }
        .links-container li:hover {
            background-color: #e3f2fd; /* 浅蓝色背景 */
        }
        .links-container a {
            text-decoration: none !important; /* 去掉链接的下划线 */
            color: #1976d2; /* 蓝色文字 */
            font-family: Arial, sans-serif; /* 字体 */
            font-size: 14px; /* 字体大小 */
            display: flex; /* 使用flex布局 */
            align-items: center; /* 垂直居中对齐 */
            height: 100%; /* 确保链接高度与父元素一致 */
        }
        .material-icons {
            font-size: 20px; /* 图标大小 */
            margin-right: 8px; /* 图标和文字间的间距 */
            text-decoration: none; /* 确保图标没有下划线 */
            color: #1976d2; /* 蓝色图标 */
        }

        /* 深色模式样式 */
        @media (prefers-color-scheme: dark) {
            .links-container li {
                background-color: #1976d2; /* 蓝色背景 */
                border-color: #1565c0; /* 加深蓝色边框 */
            }
            .links-container li:hover {
                background-color: #1565c0; /* 加深蓝色背景 */
            }
            .links-container a {
                color: #ffffff; /* 白色文字 */
            }
        }
        </style>
    </head>
    <body>
        <ul class="links-container">
            <li><a href="/"><span class="material-icons">home</span> 主页</a></li>
            <li><a href="/upload_data"><span class="material-icons">cloud_upload</span> 上传数据</a></li>
            <li><a href="/create_knowledge_base"><span class="material-icons">library_add</span> 创建知识库</a></li>
            <li><a href="/chat"><span class="material-icons">question_answer</span> 基于LlamaIndex的RAG问答</a></li>
            <li><a href="http://127.0.0.1/chat/share?shared_id=4ea39760cf0511efb57f0242ac120006&from=chat&auth=E4ZDUyMWVlM2E5YTExZjBiODMzMDI0Mm"><span class="material-icons">question_answer</span> 基于RAGFlow的RAG问答</a></li>
        </ul>
    </body>
</html>"""
