package try1;

import java.util.Arrays;

public class SteelBarCutting {
	
	private int[] price;
	public SteelBarCutting(int[] price){
		this.price = price;
	}
	
	
	//自顶向下的递归
	public int calcViaRecursion(int n){//n是钢条长度
		
		if(n==1)
			return this.price[0];
		
		int value = 0;//价值
		//分量的价值，最大值作为当前钢条的最大价值
		int[] v = new int[1+n/2];
		v[n/2] = this.price[n-1];
		for (int i = 1; i <= n/2; i++)
			v[i-1] = this.calcViaRecursion(i)+this.calcViaRecursion(n-i);
		value = Arrays.stream(v).max().getAsInt();
		return value;
	}
	
	
	//带备忘录的递归
	public int calcViaMemorandum(int n){
		int[] memo = new int[n+1];
		for (int i = 0; i < memo.length; i++) 
			memo[i] = -1;
		return this.cvm(n, memo)[n];
	}
	private int[] cvm(int n, int[] memo) {
		if (n == 1) memo[1] = this.price[0];
		if (memo[n] == -1) {
			int[] arr = new int[1+n/2];
			arr[n/2] = this.price[n-1];
			for (int i = 1; i <= n/2; i++)
				arr[i-1] = this.cvm(i, memo)[i]+this.cvm(n-i, memo)[n-i];
			memo[n] = Arrays.stream(arr).max().getAsInt();
		}
		return memo;
	}
	
	
	//自底向上的递推：随机二分
	public int calcViaB2T(int n){
		int[] r = new int[n+1];
		r[0] = 0;
		r[1] = this.price[0];
		
		int max_v = 0;
		for (int i = 2; i <= n; i++) {
			max_v = this.price[i-1];
			for (int j = 1; j <= i/2; j++) {
				int v = r[j]+r[i-j];
				if (v > max_v) max_v = v;
			}
			r[i] = max_v;
		}	
		return r[n];
	}
	
	
	//自底向上的递推：1+(n-1)
	public int calcViaB2T_2(int n){
		int[] r = new int[1+n];
		r[0] = 0;
		int max_v = 0;
		for (int i = 1; i <= n; i++) {
			max_v = this.price[i-1];
			for (int j = 1; j <= (i+1)/2; j++) {
				int v = this.price[j-1]+r[i-j];
				if (v > max_v) max_v = v;
			}
			r[i] = max_v;
		}
		return r[n];
	}
	
	
	public static void main(String[] args){
		int[] price = {1,5,8,9,10,17,17,20,24,30};
		SteelBarCutting sbc = new SteelBarCutting(price);
		for (int i = 1; i < 11; i++) {
			//int v = sbc.calcViaRecursion(i);
			//int v = sbc.calcViaMemorandum(i);
			int v = sbc.calcViaB2T_2(i);
			System.out.println("length="+i+", value="+v);
		}
	}
}
