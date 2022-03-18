#include<CL\sycl.hpp>
#include<math.h>
#include"alg_chol.h"

using namespace sycl;

Matrix Cholesky_decomposition_dpc(const Matrix& matrix, size_t shift)
{
	if (matrix.sizec() != matrix.sizer())
		exit(1);

	Matrix result(matrix);

	result[0][0] = std::sqrt(result[0][0]);

	size_t block_size = result.sizec();

	try
	{
		queue q{ host_selector() };

		buffer<type, 2> mat_buff(result.p, range<2>(block_size, block_size));

		q.submit([&](handler& cgh)
		{
			auto mat = mat_buff.get_access<access::mode::read_write>(cgh);

			cgh.parallel_for(range<1>(block_size), [=](id<1> i)
				{
					if (i.get(0) == 0)
						return;
					mat[i][0] /= mat[0][0];
				});

		}).wait();

		for (int64_t i = 1; i < block_size; i++)
		{
			type sum1 = 0.0f;
			for (int64_t j = 0; j < i; j++)
			{
				q.submit([&](handler& cgh)
				{
					auto mat = mat_buff.get_access<access::mode::read>(cgh);

					int num_work_item = q.get_device().get_info<sycl::info::device::max_compute_units>() * \
						q.get_device().get_info<sycl::info::device::native_vector_width_float>();

					buffer<type, 1> sum_buf(&sum1, 1);
					auto sum_acc = sum_buf.get_access<access::mode::write>();

					cgh.parallel_for(num_work_item, [=](auto id)
						{
							size_t global_id = id[0];
							type sub_sum = 0.0f;
							for (size_t j = global_id; j < block_size; j += num_work_item)
							{
								sub_sum += mat[i][j] * mat[i][j];
							}
							auto v = sycl::ONEAPI::atomic_ref<type,
								sycl::ONEAPI::memory_order::relaxed,
								sycl::ONEAPI::memory_scope::device,
								sycl::access::address_space::global_space>(sum_acc[0]);
							v.fetch_add(sub_sum);
						});
				});
			}

			q.wait();
			
			q.submit([&](handler& cgh)
			{
				auto mat = mat_buff.get_access<access::mode::read_write>(cgh);

				cgh.single_task([=]()
				{
					mat[i][i] = sqrt(mat[i][i] - sum1);
				});
			}).wait();

			

			q.submit([&](handler& cgh)
			{
				auto mat = mat_buff.get_access<access::mode::read_write>(cgh);

				cgh.parallel_for(range<1>(block_size - i - 1), [=](id<1> j)
					{
						type sum = 0.0f;
						for (int64_t p = 0; p < i; p++)
						{
							sum += mat[i][p] * mat[j][p];
						}
						mat[j][i] = (mat[j][i] - sum) / mat[i][i];
					});
			}).wait();
		}



		q.submit([&](handler& cgh)
		{
			auto mat = mat_buff.get_access<access::mode::write>(cgh);

			int sizec = result.sizec();
			cgh.parallel_for(range<1>(sizec - 1), [=](id<1> i)
				{
					for (int j = 0; j < sizec - i.get(0) - 1; j++)
					{
						mat[i][j + 1] = 0.0f;
					}
					//memset(&mat[i][i.get(0) + 1] , 0, sizeof(type) * (result.sizer() - i.get(0) - 1));
				});
		});
				
		
	}
	catch (exception e)
	{
		std::cout << e.what();
	}

	return result;
}



//for (int64_t i = 1; i < block_size; i++)
//{
//	type sum1 = 0.0f;
//
//
//	{
//		int num_work_item = q.get_device().get_info<sycl::info::device::max_compute_units>() * \
//			q.get_device().get_info<sycl::info::device::native_vector_width_float>();
//
//		buffer<type, 1> sum_buf(&sum1, 1);
//		auto sum_acc = sum_buf.get_access<access::mode::write>();
//
//		cgh.parallel_for(num_work_item, [=](auto id)
//			{
//				size_t global_id = id[0];
//				type sub_sum = 0.0f;
//				for (size_t j = global_id; j < block_size; j += num_work_item)
//				{
//					sub_sum += mat[i][j] * mat[i][j];
//				}
//				auto v = sycl::ONEAPI::atomic_ref<type,
//					sycl::ONEAPI::memory_order::relaxed,
//					sycl::ONEAPI::memory_scope::device,
//					sycl::access::address_space::global_space>(sum_acc[0]);
//				v.fetch_add(sub_sum);
//			});
//	}
//
//	cgh.single_task([=]()
//		{
//			mat[i][i] = sqrt(mat[i][i] - sum1);
//		});
//
//	cgh.parallel_for(range<1>(block_size - i - 1), [=](id<1> j)
//		{
//			type sum = 0.0f;
//			for (int64_t p = 0; p < i; p++)
//			{
//				sum += mat[i][p] * mat[j][p];
//			}
//			mat[j][i] = (mat[j][i] - sum) / mat[i][i];
//		});
//
//}

