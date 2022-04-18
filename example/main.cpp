#include<iostream>
#include <CL\sycl.hpp>
#include"..\\OpenAPI\Matrix.h"
#include"..\\DPC++\alg_chol.h"

constexpr int size = 10;

int main()
{
	Matrix a(size, size);
	a.fillup_rand();

	a.output();

	{
		sycl::queue q{ sycl::host_selector() };
		sycl::buffer<type, 2> buf(a.p, sycl::range<2>(size, size));
		
		print_on_device(q, buf, size);
	}
	a.output();
}