#include "cblas.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

extern int im2col_k3x3_acc(float *pSrc, float *pDst, unsigned int width, unsigned int height);

#ifdef __cplusplus
}
#endif 

static inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
  return (unsigned)(a) < (unsigned)(b);
}

void im2col_cpu(const float* data_im, const int channels,
								const int height, const int width,
								const int kernel_h, const int kernel_w,
								const int pad_h, const int pad_w,
								const int stride_h, const int stride_w,
								const int dilation_h, const int dilation_w,
								float* data_col)
{
	int channel,kernel_row,kernel_col,output_rows,output_col,output_cols;
  const int output_h = (height + 2 * pad_h -
    (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w = (width + 2 * pad_w -
    (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

  const int channel_size = height * width;

  for (channel = channels; channel > 0; channel--) {
    for (kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_h + kernel_row * dilation_h;
        for (output_rows = output_h; output_rows; output_rows--)
        {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height))
            for (output_cols = output_w; output_cols; output_cols--)
              *(data_col++) = 0;
          else
          {
            int input_col = -pad_w + kernel_col * dilation_w;
            for (output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width))
                *(data_col++) = data_im[input_row * width + input_col];
              else
                *(data_col++) = 0;
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
    data_im += channel_size;
  }
}
						
void im2col_acc(const float* data_im, const int channels,
								const int height, const int width,
								const int kernel_h, const int kernel_w,
								const int pad_h, const int pad_w,
								const int stride_h, const int stride_w,
								const int dilation_h, const int dilation_w,
								float* data_col)
{
	if ((3 == kernel_h) && (3 == kernel_w) && 
			(0 == pad_h) && (0 == pad_w) && 
			(1 == stride_h) && (1 == stride_w) && 
			(1 == dilation_h) && (1 == dilation_w))
	{
		int i;
    const int output_h = (height + 2 * 0 - (1 * (kernel_h - 1) + 1)) / 1 + 1;
    const int output_w = (width + 2 * 0 -  (1 * (kernel_w - 1) + 1)) / 1 + 1;
		unsigned int OutChannelSize = output_h*output_w*sizeof(data_im[0])*kernel_h*kernel_w;
    unsigned int InChannelSize = width*height*sizeof(data_im[0]);
		#ifdef _OPENMP
		//#pragma message "omp enable"
		#endif
		
		#pragma omp parallel for
		for(i = 0 ; i < channels; i++)
		{
				im2col_k3x3_acc((float*)(((unsigned char*)data_im)+i*InChannelSize),
												(float*)(((unsigned char*)data_col)+i*OutChannelSize),
												width,
												height);
		}
	}
	else
	{
		im2col_cpu(data_im, channels,
							 height, width,
							 kernel_h, kernel_w,
							 pad_h, pad_w,
							 stride_h, stride_w,
							 dilation_h, dilation_w,
							 data_col);
	}	
}
