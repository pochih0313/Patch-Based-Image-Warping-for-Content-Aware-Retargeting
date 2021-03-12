all:
	@g++ main.cpp -o output `pkg-config --cflags --libs opencv` -I/opt/ibm/ILOG/CPLEX_Studio_Community201/cplex/include -I/opt/ibm/ILOG/CPLEX_Studio_Community201/concert/include -DIL_STD -L/opt/ibm/ILOG/CPLEX_Studio_Community201/cplex/lib/x86-64_linux/static_pic -L/opt/ibm/ILOG/CPLEX_Studio_Community201/concert/lib/x86-64_linux/static_pic -lilocplex -lconcert -lcplex -lm -pthread
run:
	@./output
clean:
	@rm output
