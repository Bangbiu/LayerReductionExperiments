#include "stdio.h"
#include "string.h"

int startWith(char* str1, char* str2) {
	for (int i = 0; i < strlen(str2); i++) {
		if(str1[i] != str2[i]) return 0;
	}
	return 1;
}

int indexOfS(char* str1, char* str2) {
	for (int i = 0; i < strlen(str1) - strlen(str2); i++) {
		if (startWith(str1 + i, str2)) return i;
	}
	return -1;
}

int indexOfE(char* str1, char* str2) {
	int end = strlen(str2);
	for (int i = 0; i < strlen(str1) - end; i++) {
		if (startWith(str1 + i, str2)) return i + end;
	}
	return -1;
}

int substring(char* from, char* buff, int start, int end) {
	int index = 0;
	for (int i = start; i < end; i++) {
		buff[index] = from[i];
		index++;
	}
	buff[index] = '\0';
	return index;
}

int extract(char* from, char* res) {
	if (startWith(from, "[epoch ")) {
		substring(from, res, indexOfE(from, "train_loss: "), indexOfS(from, "val_accuracy: "));
		substring(from, res + strlen(res), indexOfE(from, "val_accuracy: "), strlen(from));
		return 1;
	} else if (startWith(from, "Start Training")) {
		substring(from, res, 0, strlen(from));
		return 1;
	}
	return 0;
}

int main(int argc, char *argv[]) {
	
	if (argc < 2) {
		return 0;
	}
	
	printf("%s",argv[1]);
	
	FILE *input = fopen(argv[1],"r");
	FILE *output = fopen("train_data.csv","w");
	
	char buf[255];
	
	while (fgets(buf,255,input) != NULL) {
		char extracted[255];
		if (extract(buf, extracted)) {
			fprintf(output, extracted);
		}
	}
	
	fclose(input);
	fclose(output);
}

 


