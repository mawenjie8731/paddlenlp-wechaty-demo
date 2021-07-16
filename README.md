# PaddleNLP-wechaty-demo

本例子展示一个基于 PaddleNLP + Wechaty 的微信情感识别机器人demo。通过Wechaty获取微信接收的消息，然后使用PaddleNLP的`TextCNN`模型对输入的文本进行情感判断，最终以微信消息的形式返回，实现对文本情感的识别。

更多使用PaddleNLP案例与Wechaty碰撞有趣的火花等你来探索💥客服、闲聊、对诗、写作、彩虹屁，起名、对联、码代码等你来尝试～来PaddleNLP中尽情探索吧[PaddleNLP/examples](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples)。

## 风险提示

本项目采用的api为第三方——Wechaty提供，**非微信官方api**，用户需承担来自微信方的使用风险。
在运行项目的过程中，建议尽量选用**新注册的小号**进行测试，不要用自己的常用微信号。

## Wechaty

关于Wechaty和python-wechaty，请查阅以下官方repo：

- [Wechaty](https://github.com/Wechaty/wechaty)
- [python-wechaty](https://github.com/wechaty/python-wechaty)
- [python-wechaty-getting-started](https://github.com/wechaty/python-wechaty-getting-started/blob/master/README.md)

## 环境准备

- 系统环境：Linux, MacOS, Windows
- python3.7+

## 安装和使用

1. 本项目采用的是PaddleNLP中的**TextCNN**预训练模型来完成中文对话情绪识别任务。

   详情请查阅[textcnn](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/sentiment_analysis/textcnn)。

2. Clone本项目代码

   ```
   git clone https://github.com/mawenjie8731/paddlenlp-wechaty-demo.git
   cd paddlenlp-wechaty-demo
   ```

3. 安装依赖 —— paddlepaddle, paddlenlp, wechaty

   ```
   pip install -r requirements.txt
   ```

4. 下载预训练模型和词表

   -先下载词汇表文件，用于构造词-id映射关系。

   下载链接：[https://paddlenlp.bj.bcebos.com/robot_chat_word_dict.txt](https://paddlenlp.bj.bcebos.com/robot_chat_word_dict.txt) 

   -这里我们提供了一个百度基于海量数据训练好的TextCNN模型，用户通过以下方式下载预训练模型。

   下载链接：[https://paddlenlp.bj.bcebos.com/models/textcnn.pdparams](https://paddlenlp.bj.bcebos.com/models/textcnn.pdparams)

   最后，将robot_chat_word_dict.txt和textcnn.pdparams文件放至./examples/checkpoints中。

5. Set token for your bot

   在当前系统的环境变量中，配置以下与`WECHATY_PUPPET`相关的两个变量。 关于其作用详情和TOKEN的获取方式，请查看[Wechaty Puppet Services](https://wechaty.js.org/docs/puppet-services/)。

   ```
   export WECHATY_PUPPET=wechaty-puppet-service
   export WECHATY_PUPPET_SERVICE_TOKEN=your_token_at_here
   ```

   [Paimon](https://wechaty.js.org/docs/puppet-services/paimon/)的短期TOKEN经测试可用，其他TOKEN请自行测试使用。

6. Run the bot

   ```
   python paddlenlp-chatbot.py
   ```

   运行后，可以通过微信移动端扫码登陆，登陆成功后则可正常使用。

## 运行效果

在`examples/paddlenlp-chatbot.py`中，通过以下代码即可构建一个`TextCNN`的模型，关于模型的更多用法，请查看[textcnn](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/sentiment_analysis/textcnn)。

```
# Construct the newtork.
vocab_size = len(vocab)
num_classes = len(label_map)
pad_token_id = vocab.to_indices('[PAD]')

model = TextCNNModel(
    vocab_size,
    num_classes,
    padding_idx=pad_token_id,
    ngram_filter_sizes=(1, 2, 3))

# Load model parameters.
state_dict = paddle.load('./checkpoints/textcnn.pdparams')
```

`on_message`方法是接收到消息时的回调函数，可以通过自定义的条件(譬如消息类型、消息来源、消息文字是否包含关键字、是否群聊消息等等)来判断是否回复信息，消息的更多属性和条件可以参考[Class Message](https://github.com/Wechaty/wechaty#3-class-message)。

本示例中的`on_message`方法的代码如下，脚本中回复的条件是：

1. 消息类型是文字
2. 文字信息以`[Test]`开头

```
async def on_message(msg: Message):
    """
    Message Handler for the Bot
    """
    ### PaddleNLP chatbot
    if isinstance(msg.text(), str) and len(msg.text()) > 0 \
        and msg._payload.type == MessageType.MESSAGE_TYPE_TEXT \
        and msg.text().startswith('[Test]'):  # Use a special token '[Test]' to select messages to respond.
        input_text = msg.text().replace('[Test]', '')
        examples = preprocess_prediction_data([input_text], tokenizer, pad_token_id)
        print("example:",examples)
        results = predict(
            model,
            examples,
            label_map=label_map,
            batch_size=1,
            pad_token_id=pad_token_id)

        result = results[0]
        # print('result', result)    
        await msg.say(result)  # Return the text by PaddleNLP chatbot
```

脚本成功运行后，所登陆的账号即可作为一个Chatbot，在发送需要判断情感的文本前加[Test]标识符，下图左侧的内容是Chatbot判断的结果。
![image](https://user-images.githubusercontent.com/56876519/125900114-447c6f26-dca1-48f2-ad39-5eb4ea035677.png)

