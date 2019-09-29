# include "stdio.h"
# define URL "http://www.baidu.com"
# define NUM 10

int main()
{
    printf("今天我访问这个网站：%s %d次！\n", URL, NUM);
    printf("URL's length is %d\n", sizeof(URL));

    int a;
    a = sizeof(NUM);
    printf("%d", a);

    return 0;
}