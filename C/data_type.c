# include "stdio.h"

int main()
{
    printf("int's length is %d\n", sizeof(int));
    printf("short int's length is %d\n", sizeof(short int));
    printf("long int's length is %d\n", sizeof(long int));
    printf("long long int's length is %d\n", sizeof(long long int));
    printf("char's length is %d\n", sizeof(char));
    printf("_Bool's length is %d\n", sizeof(_Bool));
    printf("float's length is %d\n", sizeof(float));
    printf("double's length is %d\n", sizeof(double));
    printf("long double's length is %d", sizeof(long double));
    return 0;
}