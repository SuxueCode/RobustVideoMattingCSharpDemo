using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using Microsoft.Win32;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using static System.Collections.Specialized.BitVector32;
using Image = SixLabors.ImageSharp.Image;
using SixLabors.ImageSharp.Processing;

namespace RobustVideoMattingCSharpDemo
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
            ONNXListInit();
        }

        private void ONNXListInit()
        {
            string path = Environment.CurrentDirectory;
            string[] files = Directory.GetFiles(path);
            foreach (string file in files)
            {
                if(Path.GetExtension(file) == ".onnx")
                {
                    cb_onnx_select.Items.Add(Path.GetFileName(file));
                }
            }
        }

        private void btn_load_img_Click(object sender, RoutedEventArgs e)
        {
            if ((string)cb_onnx_select.SelectedItem == null)
            {
                MessageBox.Show("请选择模型");
                return;
            }

            string imgPath = SelectFile();
            if (string.IsNullOrEmpty(imgPath))
            {
                MessageBox.Show("取消了选择");
                return;
            }
            img_source.Source = new BitmapImage(new Uri(imgPath));

            
            string path = Environment.CurrentDirectory + "\\" + cb_onnx_select.SelectedItem;

            
            Task.Run(() =>
            {
                using (var session = new InferenceSession(path))
                {
                    Image<Rgb24> image = Image.Load<Rgb24>(imgPath);
                    Tensor<float> input = new DenseTensor<float>(new[] { 1, 3, image.Height, image.Width });
                    image.ProcessPixelRows(accessor =>
                    {
                        for (int y = 0; y < accessor.Height; y++)
                        {
                            Span<Rgb24> pixelSpan = accessor.GetRowSpan(y);
                            for (int x = 0; x < accessor.Width; x++)
                            {
                                input[0, 0, y, x] = pixelSpan[x].R / 255f;
                                input[0, 1, y, x] = pixelSpan[x].G / 255f;
                                input[0, 2, y, x] = pixelSpan[x].B / 255f;
                            }
                        }
                    });

                    int image_pixel = image.Height * image.Width;
                    float ratio = 0f;
                    if (image_pixel < 512 * 512)
                    {
                        ratio = 1f;
                    }
                    else if (image_pixel < 1920 * 1080)
                    {
                        ratio = 0.6f;
                    }
                    else if (image_pixel < 3840 * 2160)
                    {
                        ratio = 0.4f;
                    }
                    else
                    {
                        ratio = 0.2f;
                    }
                    Tensor<float> r1i = new DenseTensor<float>(new[] { 1, 1, 1, 1 });
                    Tensor<float> r2i = new DenseTensor<float>(new[] { 1, 1, 1, 1 });
                    Tensor<float> r3i = new DenseTensor<float>(new[] { 1, 1, 1, 1 });
                    Tensor<float> r4i = new DenseTensor<float>(new[] { 1, 1, 1, 1 });
                    Tensor<float> downsample_ratio = new DenseTensor<float>(new[] { ratio }, new int[] { 1 });
                    var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("src", input),
                    NamedOnnxValue.CreateFromTensor("r1i", r1i),
                    NamedOnnxValue.CreateFromTensor("r2i", r2i),
                    NamedOnnxValue.CreateFromTensor("r3i", r3i),
                    NamedOnnxValue.CreateFromTensor("r4i", r4i),
                    NamedOnnxValue.CreateFromTensor("downsample_ratio", downsample_ratio),
                };

                    var outputs = session.Run(inputs);

                    var fgr = outputs.Single(x => x.Name == "fgr").AsTensor<float>();
                    var pha = outputs.Single(x => x.Name == "pha").AsTensor<float>();

                    Image<Argb32> outImg = new Image<Argb32>(image.Width, image.Height);
                    Image<Argb32> backImg = new Image<Argb32>(image.Width, image.Height);
                    backImg.ProcessPixelRows(accessor =>
                    {
                        for (int y = 0; y < accessor.Height; y++)
                        {
                            Span<Argb32> pixelSpan = accessor.GetRowSpan(y);
                            for (int x = 0; x < accessor.Width; x++)
                            {
                                pixelSpan[x].A = (byte)(255f);
                                pixelSpan[x].R = (byte)(255f);
                                pixelSpan[x].G = (byte)(255f);
                                pixelSpan[x].B = (byte)(255f);
                            }
                        }
                    });
                    int pha_pixel = 0;
                    outImg.ProcessPixelRows(accessor =>
                    {
                        for (int y = 0; y < accessor.Height; y++)
                        {
                            Span<Argb32> pixelSpan = accessor.GetRowSpan(y);
                            for (int x = 0; x < accessor.Width; x++)
                            {
                                pixelSpan[x].A = (byte)(pha[0, 0, y, x] * 255f);
                                if(pixelSpan[x].A != 0)
                                {
                                    pixelSpan[x].R = (byte)((fgr[0, 0, y, x] * 255f) > 255 ? 255 : (fgr[0, 0, y, x] * 255f));
                                    pixelSpan[x].G = (byte)((fgr[0, 1, y, x] * 255f) > 255 ? 255 : (fgr[0, 1, y, x] * 255f));
                                    pixelSpan[x].B = (byte)((fgr[0, 2, y, x] * 255f) > 255 ? 255 : (fgr[0, 2, y, x] * 255f));
                                }
                                pha_pixel++;
                            }
                        }
                    });

                    backImg.Mutate(a => a.DrawImage(outImg, 1));
                    string savePath = imgPath + "_1.jpg";
                    backImg.SaveAsJpeg(savePath);
                    backImg.Dispose();

                    Dispatcher.Invoke(() =>
                    {
                        img_process_source.Source = new BitmapImage(new Uri(savePath));
                    });
                }
            });
        }
        private static string SelectFile()
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.Title = "jpg";
            openFileDialog.Filter = "jpg 文件 (*.jpg)|*.jpg|所有文件 (*.*)|*.*";
            // 设置默认打开的文件夹为桌面
            openFileDialog.InitialDirectory = Environment.GetFolderPath(Environment.SpecialFolder.Desktop);
            if (openFileDialog.ShowDialog() == true)
            {
                return openFileDialog.FileName;
            }
            else
            {
                return "";
            }
        }
    }
}
