void BitSet(uint x, uint y, uint pw, __global uint* pattern) { 
	pattern[y * pw + (x >> 5)] |= 1U << (int)(x & 31); 
}

uint GetBit(uint x, uint y, uint pw, __global uint* second) {
 
	return (second[y * pw + (x >> 5)] >> (int)(x & 31)) & 1U; 
}

__kernel void empty_patterns (__global uint* pattern)
{
    int i = get_global_id(0);
    pattern[i] = 0;
}

__kernel void swap_buffers (__global uint* pattern, __global uint* second)
{
    int i = get_global_id(0); //work item ID
	second[i] = pattern [i];
}

__kernel void process_pixels (__global uint* pattern, __global uint* second, uint pw)
{
    uint x = get_global_id(0);
	uint y = get_global_id(1);

	if (x != 0 && y != 0) {
		uint n = GetBit(x - 1, y - 1, pw, second) + GetBit(x, y - 1, pw, second) + GetBit(x + 1, y - 1, pw, second) + GetBit(x - 1, y, pw, second)
			   + GetBit(x + 1, y, pw, second) + GetBit(x - 1, y + 1, pw, second) + GetBit(x, y + 1, pw, second) + GetBit(x + 1, y + 1, pw, second);
	
		if ((GetBit(x, y, pw, second) == 1 && n == 2) || n == 3) {
			BitSet(x, y, pw, pattern);
		}
	}
}