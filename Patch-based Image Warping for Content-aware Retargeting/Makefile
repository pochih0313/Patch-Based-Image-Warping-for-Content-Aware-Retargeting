all:
	@g++ main.cpp -o output `pkg-config --cflags --libs opencv` -I/opt/ibm/ILOG/CPLEX_Studio1271/cplex/include -I/opt/ibm/ILOG/CPLEX_Studio1271/concert/include -DIL_STD -L/opt/ibm/ILOG/CPLEX_Studio1271/cplex/lib/x86-64_linux/static_pic -L/opt/ibm/ILOG/CPLEX_Studio1271/concert/lib/x86-64_linux/static_pic -lilocplex -lconcert -lcplex -lm -pthread -ldl
run:
	@./output
clean:
	@rm output
