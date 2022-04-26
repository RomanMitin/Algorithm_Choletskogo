#include<CL\sycl.hpp>
#include<math.h>
#include"alg_chol.h"

using namespace sycl;


void print_on_device(queue& q, buffer<type, 2>& data_buf, size_t size)
{
	q.submit([&](handler& cgh)
		{
			auto data = data_buf.get_access<access::mode::read>(cgh);
			stream ostream(1024, 768, cgh);
			cgh.single_task([=]()
				{
					ostream << hex;
					/*ostream << scientific;
					ostream << setprecision(16);*/
					for (size_t i = 0; i < size; i++)
					{
						for (size_t j = 0; j < size; j++)
						{
							ostream << *(uint32_t*)&data[i][j] << '\t';
						}
						ostream << endl;
					}
					ostream << endl;
				});
		});
}

void print_on_device2(queue& q, buffer<type, 1>& data_buf, size_t size)
{
	q.submit([&](handler& cgh)
		{
			auto data = data_buf.get_access<access::mode::read>(cgh);
			stream ostream(1024, 80, cgh);
			cgh.single_task([=]()
				{
					ostream << scientific;
					ostream << setprecision(10);
					for (size_t i = 0; i < size; i++)
					{
						for (size_t j = 0; j < size; j++)
						{
							ostream << data[i * size + j] << '\t';
						}
						ostream << endl;
					}
					ostream << endl;
				});
		});
}

Matrix Cholesky_decomposition_dpc(const Matrix& matrix)
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

		//print_on_device(q, mat_buff, block_size);

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

		//print_on_device(q, mat_buff);

		size_t num_work_items = q.get_device().get_info<sycl::info::device::max_compute_units>();
		type* tmp_sum = new type[num_work_items];
		type* sum1 = new type;

		for (size_t i = 0; i < num_work_items; i++)
			tmp_sum[i] = 0.0f;
		*sum1 = 0.0f;

		{
			buffer<type, 1> accum_buf(tmp_sum, range<1>(num_work_items));
			buffer<type, 1> sum_buf(sum1, range<1>(1));

			for (int64_t i = 1; i < block_size; i++)
			{
				/*if (i == 2)
					print_on_device(q, mat_buff, block_size);*/

				q.submit([&](handler& cgh) {
					auto mat_acc = mat_buff.get_access<access::mode::read>(cgh);
					auto accum_acc = accum_buf.get_access<access::mode::write>(cgh);

					stream ostream(1024, 80, cgh);

					cgh.parallel_for(num_work_items, [=](id<1> index) {
						size_t glob_id = index.get(0);
						type sum = 0.0f;
						for (size_t j = glob_id; j < i; j += num_work_items)
						{
							sum += mat_acc[i][j] * mat_acc[i][j];
							/*if (i == 4)
								ostream << "glob_id: " << glob_id << " mat[" << i << "][" << j << "] " << mat_acc[i][j] << '\n';*/
						}
						accum_acc[glob_id] = sum;

						/*if (i == 4)
							ostream << "glob_id: " << glob_id << " sum: " << sum << '\n';*/
						});
					}).wait();

				q.submit([&](handler& cgh)
					{
						auto accum_acc = accum_buf.get_access<access::mode::read_write>(cgh);
						auto sum = sum_buf.get_access<access::mode::read_write>(cgh);
						auto mat = mat_buff.get_access<access::mode::read_write>(cgh);

						stream ostream(1024, 80, cgh);

						cgh.single_task([=]()
							{
								for (size_t j = 0; j < i && j < num_work_items; ++j) // поправил цикл
								{
									sum[0] += accum_acc[j];
									accum_acc[j] = 0.0f;
								}

								mat[i][i] = sqrt(mat[i][i] - sum[0]);
								//ostream << sum[0] << '\n';
								sum[0] = 0.0f;
								
							});
					}).wait();



					q.submit([&](handler& cgh)
						{
							auto mat = mat_buff.get_access<access::mode::read_write>(cgh);

							cgh.parallel_for(range<1>(block_size - i - 1), [=](id<1> j)
								{
									j += i + 1;
									type sum = 0.0f;
									for (int64_t p = 0; p < i; p++)
									{
										sum += mat[i][p] * mat[j][p];
									}
									mat[j][i] = (mat[j][i] - sum) / mat[i][i];
								});
						}).wait();
			}
		}
		delete[] tmp_sum;
		delete sum1;

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
	catch (exception e)
	{
		std::cout << e.what();
	}

	return result;
}

void Cholesky_decomposition_dpc(queue& q, buffer<type, 2>& mat_buff, size_t shift, size_t block_size, type* tmp_sum, type* sum1)
{

	size_t num_work_items = q.get_device().get_info<sycl::info::device::max_compute_units>();
	buffer<type, 1> accum_buf(tmp_sum, range<1>(num_work_items));
	buffer<type, 1> sum_buf(sum1, range<1>(1));

	q.submit([&](handler& cgh)
		{
			auto accum_acc = accum_buf.get_access<access::mode::write>(cgh);
			auto sum = sum_buf.get_access<access::mode::write>(cgh);
			auto mat = mat_buff.get_access<access::mode::read_write>(cgh);

			cgh.single_task([=]()
				{
					for (size_t i = 0; i < num_work_items; i++)
						accum_acc[i] = 0.0f;
					sum[0] = 0.0f;
					mat[shift][shift] = sycl::sqrt(mat[shift][shift]);
				});
		}).wait();

	q.submit([&](handler& cgh)
		{
			auto mat = mat_buff.get_access<access::mode::read_write>(cgh);
		
			cgh.parallel_for(range<1>(block_size) - 1, [=](id<1> i)
				{
					mat[i + 1 + shift][shift] /= mat[shift][shift];
	
				});

		}).wait();

	//print_on_device(q, mat_buff, 10);
	for (int64_t i = 1; i < block_size; i++)
	{
		/*if (i == 2)
			print_on_device(q, mat_buff, block_size);*/

		q.submit([&](handler& cgh) {
			auto mat_acc = mat_buff.get_access<access::mode::read>(cgh);
			auto accum_acc = accum_buf.get_access<access::mode::write>(cgh);

			stream ostream(1024, 80, cgh);

			cgh.parallel_for(num_work_items, [=](id<1> index) {
				size_t glob_id = index.get(0);
				type sum = 0.0f;
				for (size_t j = glob_id; j < i; j += num_work_items)
				{
					sum += mat_acc[i + shift][j + shift] * mat_acc[i + shift][j + shift];
					/*if (i == 4)
						ostream << "glob_id: " << glob_id << " mat[" << i << "][" << j << "] " << mat_acc[i][j] << '\n';*/
				}
				accum_acc[glob_id] = sum;

				/*if (i == 4)
					ostream << "glob_id: " << glob_id << " sum: " << sum << '\n';*/
				});
			}).wait();

		q.submit([&](handler& cgh)
			{
				auto accum_acc = accum_buf.get_access<access::mode::read_write>(cgh);
				auto sum = sum_buf.get_access<access::mode::read_write>(cgh);
				auto mat = mat_buff.get_access<access::mode::read_write>(cgh);

				stream ostream(1024, 80, cgh);

				cgh.single_task([=]()
					{
						for (size_t j = 0; j < num_work_items; ++j) // поправил цикл
						{
							sum[0] += accum_acc[j];
							accum_acc[j] = 0.0f;
						}

						mat[i + shift][i + shift] = sqrt(mat[i + shift][i + shift] - sum[0]);
						//ostream << sum[0] << '\n';
						sum[0] = 0.0f;

					});
			}).wait();

		q.submit([&](handler& cgh)
			{
				auto mat = mat_buff.get_access<access::mode::read_write>(cgh);

				cgh.parallel_for(range<1>(block_size - i - 1), [=](id<1> j)
					{
						j += i + 1;
						type sum = 0.0f;
						for (int64_t p = 0; p < i; p++)
						{
							sum += mat[i + shift][p + shift] * mat[j + shift][p + shift];
						}
						mat[j + shift][i + shift] = (mat[j + shift][i + shift] - sum) / mat[i + shift][i + shift];
					});
			}).wait();
	}
}

