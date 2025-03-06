
/* number of elements in array equal to z */
int num_eq(array[] int xs, int z) {
    int k = 0;
    for ( x in xs ) {
        k += (x == z);
    }
    return k;
}

/* return indices of array with elements equal to z */
array[] int indices_eq(array[] int xs, int z) {
    int n = num_elements(xs);
    int m = num_eq(xs, z);
    array[m] int ys;
    int pos = 1;
    for ( idx in 1:n ) {
        if ( xs[idx] == z ) {
            ys[pos] = idx;
            pos += 1;
        }
    }
    return ys;
}
