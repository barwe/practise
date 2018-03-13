public class eightQueen {

	private int numQ; //皇后数量

	//构造函数
	public eightQueen() {numQ = 8;}
	public eightQueen(int numQueen) {numQ = numQueen;}

	//检查某个位置上方是否存在queen
	public boolean check(int[][] arr, int x, int y){
		if (arr[x][y] == 1) return false;
		for (int i = 0; i < x; i++) {
			//media
			if (arr[i][y] == 1) return false;
			//left
			if (y-x+i >= 0 && arr[i][y-x+i] == 1) return false;
			//right
			if (y+x-i < numQ && arr[i][y+x-i] == 1) return false;
		}
		return true;
	}
	
	/*在给定棋盘的某一行进行搜索：
	 * 1.当前位置合适：暂停搜索本行，调到下一行进行搜索
	 * 2.当前位置不合适：横向移动到下一位置
	 * 3.当前行均不合适：返回false表示结束本次深度优先搜索
	 * 4.当前搜索行不存在时直接返回false结束本次深度优先搜索
	 * */
	public boolean searchLine(int[][] arr, int x){
		
		//行索引不存在时结束搜索
		if (x == numQ) {
			printChessboard(arr);
			return true;
		}
		
		//遍历当前行的所有点后仍为true将直接结束本次深度搜索
		boolean no_good = true;
		
		//遍历当前行的所有点
		for (int i = 0; i < numQ; i++) {
			
			//当当前点可以放置皇后时
			if (check(arr, x, i) == true) {
				//设为1表示已放置皇后
				arr[x][i] = 1;
				//暂停搜索当前行的剩余位置，搜索下一行是否存在合适位置
				//如果下一行存在合适位置优先对下一行进行搜索
				if (searchLine(arr, x+1) == true) {
					//此时当前行已经存在合适位置，no_good应该设为false
					no_good = false;
					//在继续搜索剩余位置时要将上面已经设置过皇后的位置还原
					arr[x][i] = 0;
					//继续搜索当前行的剩余位置
					continue;
				}
				//不管中间发生了什么保证将可以设为皇后的这个位置还原
				arr[x][i] = 0;
			}
			//当当前位置不能放置皇后时继续横向遍历
			else continue;
		}
		
		//如果当前行所有位置均不符合要求直接结束后续搜索，提供true值供上一行继续搜索
		if (no_good == true) 
			return true;	
		
		//当前行存在可以放置皇后的位置时返回false
		return false;
	}
	
	//打印二维数组
 	public void printChessboard(int[][] arr2d){
		for (int i = 0; i < numQ; i++) {
			for (int j = 0; j < numQ; j++) {
				System.out.print(arr2d[i][j]);
				System.out.print(' ');
			}
			System.out.println();
		}
		System.out.println("---------------");
	}
	 
 	//定义二维数组
 	public int[][] getArr() {
		int[][] arr = new int[numQ][numQ];
		for (int i = 0; i < numQ; i++) {
			for (int j = 0; j < numQ; j++) {
				arr[i][j] = 0;
			}
		}
		return arr;
	}
 	
	public static void main(String[] args) {
		eightQueen eg = new eightQueen(16);
		int[][] cb = eg.getArr();
		eg.searchLine(cb, 0);
		
	}
}
