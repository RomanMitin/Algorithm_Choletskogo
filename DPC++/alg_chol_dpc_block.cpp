#include <CL/sycl.hpp>
#include "alg_chol.h"

using namespace sycl;

void findL21(queue& q, buffer<type, 2>& mat_buff, size_t shift, size_t block_size, size_t size)
{
	for (size_t i = 0; i < block_size; i++)
	{
		q.submit([&](handler& cgh)
			{
				auto mat_acc = mat_buff.get_access<access::mode::read_write>(cgh); // Будет параллелиться?

				cgh.parallel_for(range<1>(size - block_size - shift), [=](id<1> item)
					{
						size_t j = item.get(0);
						type tmp = mat_acc[j + block_size + shift][i + shift];
						type sum = 0;

						for (size_t k = 0; k < i; k++)
						{
							sum += mat_acc[i + shift][k + shift] * mat_acc[j + block_size + shift][k + shift];
						}
						mat_acc[j + block_size + shift][i + shift] = tmp - sum;

						mat_acc[j + block_size + shift][i + shift] /= mat_acc[i + shift][i + shift];
					});
			});
	}
	q.wait();
}

void findA22(queue& q, buffer<type, 2>& mat_buff, size_t shift, size_t block_size, size_t mat_block)
{
	size_t row = mat_buff.get_size() - shift - block_size;
	size_t col = block_size;

	q.submit([&](handler& cgh)
		{
			auto mat_acc = mat_buff.get_access<access::mode::write>(cgh);

			cgh.parallel_for(nd_range<2>(range<2>(row, col), range<2>(mat_block, mat_block)), [=](nd_item<2> item)
				{
					type sum = 0.0f;
					size_t i = item.get_global_id(0);
					size_t j = item.get_global_id(1);
					for (size_t k = 0; k < col; k++)
					{
						sum += mat_acc[i + block_size + shift][k + shift] * mat_acc[j + block_size + shift][k + shift];
					}
					mat_acc[i + block_size + shift][j + block_size + shift] -= sum;
				});

		});
}

Matrix Cholesky_decomposition_dpc_block(const Matrix& mat, size_t block_size)
{
	Matrix result(mat);

	queue q{ host_selector() };

	int64_t shift;
	size_t num_work_items = q.get_device().get_info<sycl::info::device::max_compute_units>();
	type* tmp_sum = new type[num_work_items];
	type* sum1 = new type;

	size_t mat_size = result.sizec();
	
	{
		buffer<type, 2> mat_buff(result.p, range<2>(mat_size, mat_size));
		for (shift = 0; shift < int64_t(result.sizer() - block_size); shift += block_size)
		{
			// Обычный алгоритм Холетского для блока L11
			Cholesky_decomposition_dpc(q, mat_buff, shift, block_size, tmp_sum, sum1);

			// Решение нижнетреугольной системы для нахождения блока L21
			findL21(q, mat_buff, shift, block_size, mat.sizec());

			// Нахождение редуцированной матрицы A22 с помощью вычитания из исходной матрицы A22 блока L21 "В квадрате"
			findA22(q, mat_buff, shift, block_size, 1);
		}

		
		//Обычный алгоритм Холетского для последнего блока 
		block_size = result.sizer() - shift;
		Cholesky_decomposition_dpc(q, mat_buff, shift, block_size, tmp_sum, sum1);
		delete[] tmp_sum;
		delete sum1;


		//Зануление верхней части матрицы
		q.submit([&](handler& cgh)
			{
				auto mat = mat_buff.get_access<access::mode::write>(cgh);

				int sizec = result.sizec();
				cgh.parallel_for(range<1>(sizec - 1), [=](id<1> i)
					{
						for (int j = i + 1; j < sizec; j++)
						{
							mat[i][j] = 0.0f;
						}
					});
			});

	}
	

	return result;
}
//void _block_mm(sycl::queue queue, std::vector<float>& a, std::vector<float>& b, \
//	std::vector<float>& c, uint32_t size, uint32_t block) {
//
//	sycl::buffer<float, 1> buf_c(c.data(), c.size());
//	sycl::buffer<float, 1> buf_a(a.data(), a.size());
//	sycl::buffer<float, 1> buf_b(b.data(), b.size());
//
//	queue.submit([&](handler& cgh) {
//
//		sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> \
//			buffer(2 * block * block, cgh);
//
//		auto in_c = buf_c.get_access<sycl::access::mode::write>(cgh);
//		auto a = buf_a.get_access<sycl::access::mode::read>(cgh);
//		auto b = buf_b.get_access<sycl::access::mode::read>(cgh);
//
//		cgh.parallel_for<class _Mult>(nd_range<2>(range<2>(size, size), range<2>(block, block)), [=](nd_item<2> item) {
//			float* block_a = buffer.get_pointer();
//			float* block_b = block_a + block * block;
//
//			size_t li = item.get_local_id(0);	//локальный индекс в группе (строка)
//			size_t lj = item.get_local_id(1);
//			uint32_t gi = block * item.get_group(0) + li;	//начало номера группы по строке 
//			uint32_t gj = block * item.get_group(1) + lj;
//			uint32_t block_count = size / block;
//
//			float sum = 0.0f;
//			for (size_t i = 0; i < block_count; ++i) {
//				block_a[li * block + lj] = a[gi * size + block * i + lj];
//				block_b[li * block + lj] = b[(block * i + li) * size + gj];
//				item.barrier(sycl::access::fence_space::local_space);
//				for (int k = 0; k < block; ++k) {
//					sum += block_a[li * block + k] * block_b[k * block + lj];
//				}
//				item.barrier(sycl::access::fence_space::local_space);
//			}
//			in_c[gi * size + gj] = sum;
//
//			});
//		}).wait();
//}