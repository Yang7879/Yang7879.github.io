'''
    Time: 2020.01.08 20:25  -- by Juice @ IIAU
    visualization可视化
'''
import visdom
import numpy as np


class Visdom_Visualization(object):
    def __init__(self, env_forename, window_titles):

        # 定义三个绘图环境，第一个用来画曲线，第二个用来显示训练集中的某些图片和分割结果，第三个用来显示validation集的图片
        self.vis_for_line = visdom.Visdom(use_incoming_socket=False,env=env_forename + '-' + 'main')
        self.vis_for_train_imgs = visdom.Visdom(env=env_forename + '-' + 'train-imgs')
        self.vis_for_valid_imgs = visdom.Visdom(env=env_forename + '-' + 'valid-imgs')
        
        # 因为第一个画曲线的环境要画很多条线，所以给每个曲线定义一个窗口，把窗口实例放在window_list里面，到时候根据编号引用
        self.window_list = []
        self.window_titles = window_titles
        for i in range(len(self.window_titles)):
            self.window_list.append(self.vis_for_line.line(X=np.array([0]), Y=np.array([0]), opts=dict(title=self.window_titles[i])))

        # print(self.window_titles.__len__())

    def update_line(self, window_id: str, x, y):
        '''
        注意这是一个画线的函数，所有线都在self.vis_for_line 这个环境里面
        :param window_id: 指定要在哪个窗口里面画线，给一个字符串，这个字符串是窗口的标题
        :param x: x轴坐标，我只试过用整数，这个形参我一般给当前迭代到多少步了
                （如果每个epoch画一次，x就给epoch序号，如果每100个iteration画一次，那就每100个iteration调用一次这个函数，x值给当前iter的次数）
        :param y: y轴坐标，给一个数就行，小数整数都可以
        :return:
        '''
        if isinstance(window_id, str):  # 把窗口的名字转换成window_titles这个list的下标
            window_id = self.window_titles.index(window_id)

        self.vis_for_line.line(X=np.array([x]), Y=np.array([y]), win=self.window_list[window_id], update='append')

    def show_images(self, env, imgs):
        '''
        这个函数用来显示图片
        :param env:指示在哪个环境里画图，只能是字符串，'train'或者'valid'，分别对应初始化时建立的第2、3个环境
        :param imgs:要显示的图片，只能是Pytorch的Tensor。形状可以是下面几种，可能不全，可以多试试：
                    1.一个batch的RGB图像： 形状应该是[batchsize, 3, height, width]
                    2.一个batch的灰度图像： 形状应该是[batchsize, 1, height, width]
                    3.一张RGB图像： 形状应该是[1, 3, height, width]或者[3, height, width]
                    4.一张一通道图像： 形状应该是[1, 1, height, width]或者[1, height, width]
                    能不能画形状是[height, width]的图像不太确定了。我现在会用for循环把一个batch里面的n张图分开来画，多次调用这个函数，每次只传入情况3或者情况4 形状的数据

                    除了上面说的形状的问题之外，还有数值范围的问题，根据我用的经验：
                    如果输入的imgs取值范围在0-1之内的，它会按照最大值是1来画图，一个例子是：把RGB图像除以255之后所得的数据直接传进来，可以正常显示色彩
                    如果输入的imgs里面有的数据大于1，那它会按照最大值是255来画图，超过255的咋处理这个我没注意过
        :return:
        '''

        if env == "train":
            self.vis_for_train_imgs.images(imgs)
        elif env == "valid":
            self.vis_for_valid_imgs.images(imgs)



if __name__ == "__main__":
    window_titles = [
        "learning_rate",
        "train_loss_epoch",
        "valid_loss_epoch",
        "train_batch_loss",

        "1_classification_loss",
        "2_regression_loss",
        "3_mse_loss_2",
        "4_mse_loss_3"
    ]

    vis = Visdom_Visualization(env_forename='model1', window_titles=window_titles)

    #
    # if (iter_num < 500 and batch % 10 == 0) or (iter_num > 500 and batch % 100 == 0):
    #     vis.update_line("train_batch_loss", iter_num, batch_loss.item())
    #
    #     vis.update_line("1_loss_fused", iter_num, loss_fused.item())
    #     vis.update_line("2_loss_local", iter_num, loss_local.item())
    #     vis.update_line("3_loss_global", iter_num, loss_global.item())
    #     vis.update_line("4_loss_guidance", iter_num, loss_guidance.item())
    #
    # if batch % (num_train_samples // 6) == 0:
    #     for i in range(gt_depth.size()[0]):
    #         vis.show_images("train", input[i, 1:])
    #         vis.show_images("train", input[i, 0:1])
    #         vis.show_images("train", global_depth[i])
    #         vis.show_images("train", local_depth[i])
    #         vis.show_images("train", fused_out[i])
    #         vis.show_images("train", gt_depth[i])

