#pragma once
#include <CL\sycl.hpp>
#include"..\\OpenAPI\Matrix.h"



void print_on_device(sycl::queue& q, sycl::buffer<type, 2>& data_buf, size_t size);
void print_on_device2(sycl::queue& q, sycl::buffer<type, 1>& data_buf, size_t size);

Matrix Cholesky_decomposition_dpc(const Matrix& mat);

Matrix Cholesky_decomposition_dpc_block(const Matrix& mat, size_t block_size = 128);

void Cholesky_decomposition_dpc(sycl::queue& q, sycl::buffer<type, 2>& mat_buff, size_t shift, size_t block_size, type* tmp_sum, type* sum1);
