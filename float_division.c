#include <stdio.h>

int main() {
    size_t granularity = 2097152;
    float size = (float)(4 * 512 * 512 + granularity -1);

    printf("%f\n", size);
}