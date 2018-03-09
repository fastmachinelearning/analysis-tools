#include <iostream>
#include <string>
#include <stdio.h>
#include <vector>

using namespace std;

int main(int argc, char* argv[])
{
  if(argc != 3)
    {
      cout << "need target FPGA and precision strings." << endl;
      return -1;
    }

  string fpga = argv[1];
  string precision = argv[2];

  cout << "fpga = " << fpga << endl;
  cout << "precision = " << precision << endl;

  unsigned int bitSize, fixPoint;
  sscanf(argv[2],"<%u,%u>",&bitSize,&fixPoint);

  cout << "bitSize = " << bitSize << endl;
  cout << "fixPoint = " << fixPoint << endl;

  unsigned int i = fpga.length()-2;
  while(i >= 0)
    {
      //cout << i << " " << fpga[i] << endl;
      if(fpga[i] != '-') break;
      i-=2;
    }
  
  string pkg = "";
  for(unsigned int j=0;j<i+2;++j)
    if(fpga[j] != '-')
      pkg += fpga[j];
  pkg += "pkg.txt";
  
  cout << "package file = " << pkg << endl;
  
  FILE *fp = fopen(("../usaall/" + pkg).c_str(),"r");
  if(!fp)
    {
      cout << "Invalid file path " << endl;
      return -1;
    }

  char line[500];
  i = 0;

  char pin[10];
  char type[100];
  char name[100];

  vector<string> pins;
  vector<string> types;
  
  unsigned int pinCnt = 0;
  while(fgets(line,500,fp))
    {
      if(i++<3) continue; //skip headers
      //cout << line;

      sscanf(line,"%s %s %*s %*s %s",pin,name,type);

      if(type[0] != 'H' || name[0] != 'I')
	continue;

      //cout << "pin " << pin << " type " << type << endl;
      pins.push_back(string(pin));
      types.push_back(string(type));
      ++pinCnt;
    }

  fclose(fp);

  cout << "pinCnt = " << pinCnt << endl;

  const unsigned int inputCnt = 16;
  const unsigned int outputCnt = 5;
  const unsigned int extraInCnt = 4;
  const unsigned int extraOutCnt = 38;
  
  cout << "Need pin count = " << inputCnt*bitSize + outputCnt*bitSize + 
    extraInCnt + extraOutCnt << endl;
  if(pinCnt < inputCnt*bitSize + outputCnt*bitSize + 
     extraInCnt + extraOutCnt)
    {
      cout << "Not enough pins!" << endl;
      return -1;
    }

  fp = fopen("barebones_top.xdc","w");
  if(!fp)
    {
      cout << "Invalid file path " << endl;
      return -1;
    }

  //handle input pins
  pinCnt = 0;

  fprintf(fp,"set_property CLOCK_DEDICATED_ROUTE FALSE [get_nets IBUF_CLK/O]\n");
  fprintf(fp,"set_max_delay 5000.00 -to [get_cells * -hierarchical -filter {IS_PRIMITIVE == true && (NAME =~ *INPUT_SIG*)}]\n");

  fprintf(fp,"create_clock -name clk -period 7 -waveform {0 3.0} [get_ports EXTRA_INPUT_PADS[0]]\n\n\n");

  for(i=0;i< extraInCnt; ++i, ++pinCnt)
    {
      fprintf(fp,"set_property PACKAGE_PIN %s [get_ports EXTRA_INPUT_PADS[%d]]\n",
	      pins[pinCnt].c_str(),i);
      fprintf(fp,"set_property IOSTANDARD LVCMOS18 [get_ports EXTRA_INPUT_PADS[%d]]\n", i);
    }
  for(i=0;i< inputCnt*bitSize; ++i, ++pinCnt)
    {
      fprintf(fp,"set_property PACKAGE_PIN %s [get_ports INPUT_PADS[%d]]\n",
	      pins[pinCnt].c_str(),i);
      fprintf(fp,"set_property IOSTANDARD LVCMOS18 [get_ports INPUT_PADS[%d]]\n", i);
    }
  for(i=0;i< extraOutCnt; ++i, ++pinCnt)
    {
      fprintf(fp,"set_property PACKAGE_PIN %s [get_ports EXTRA_OUTPUT_PADS[%d]]\n",
	      pins[pinCnt].c_str(),i);
      fprintf(fp,"set_property IOSTANDARD LVCMOS18 [get_ports EXTRA_OUTPUT_PADS[%d]]\n", i);
    }
  for(i=0;i< outputCnt*bitSize; ++i, ++pinCnt)
    {
      fprintf(fp,"set_property PACKAGE_PIN %s [get_ports OUTPUT_PADS[%d]]\n",
	      pins[pinCnt].c_str(),i);
      fprintf(fp,"set_property IOSTANDARD LVCMOS18 [get_ports OUTPUT_PADS[%d]]\n", i);
    }
  fclose(fp);


  fp = fopen("barebones_top.vhd","w");
  if(!fp)
    {
      cout << "Invalid file path " << endl;
      return -1;
    }

  fprintf(fp,"\
library ieee;\n\
use ieee.std_logic_1164.ALL;\n\
use ieee.numeric_std.ALL;\n\
use ieee.std_logic_misc.ALL;\n\
\n\
library UNISIM;\n\
use UNISIM.Vcomponents.ALL;\n\
\n\
\n\
entity top is\n\
   port ( \n\
EXTRA_INPUT_PADS : in std_logic_vector(%d downto 0);\n\
INPUT_PADS : in std_logic_vector(%d downto 0);\n\
EXTRA_OUTPUT_PADS : out std_logic_vector(%d downto 0);\n\
OUTPUT_PADS : out std_logic_vector(%d downto 0)\n\
\n\
);\n\
end top;\n\
  \n\
architecture BEHAVIORAL of top is\n\
\n\
\n\
signal clk : std_logic; \n\
signal EXTRA_INPUT_SIG : std_logic_vector(%d downto 0);\n\
signal INPUT_SIG : std_logic_vector(%d downto 0);\n\
signal EXTRA_OUTPUT_SIG : std_logic_vector(%d downto 0);\n\
signal OUTPUT_SIG : std_logic_vector(%d downto 0);\n\
\n\
signal EXTRA_INPUT_SIG2 : std_logic_vector(%d downto 0);\n\
signal INPUT_SIG2 : std_logic_vector(%d downto 0);\n\
\n\
signal EXTRA_INPUT_SIG3 : std_logic_vector(%d downto 0);\n\
signal INPUT_SIG3 : std_logic_vector(%d downto 0);\n\
\n\
\n\
begin \n\
\n\
\n\
algo : entity work.myproject_prj_1_0\n\
  port map (\n\
ap_clk => clk,\n\
ap_rst => EXTRA_INPUT_SIG(1),   \n\
ap_start => EXTRA_INPUT_SIG(2),   \n\
data_V_ap_vld => EXTRA_INPUT_SIG(3), \n\
\n\
ap_done => EXTRA_OUTPUT_SIG(0), \n\
ap_idle => EXTRA_OUTPUT_SIG(1), \n\
ap_ready => EXTRA_OUTPUT_SIG(2), \n\
res_V_ap_vld => EXTRA_OUTPUT_SIG(3), \n\
const_size_in_ap_vld => EXTRA_OUTPUT_SIG(4), \n\
const_size_out_ap_vld => EXTRA_OUTPUT_SIG(5), \n\
const_size_in => EXTRA_OUTPUT_SIG(21 downto 6), \n\
const_size_out => EXTRA_OUTPUT_SIG(37 downto 22), \n\
\n\
data_V => INPUT_SIG(%d downto 0), \n\
res_V => OUTPUT_SIG(%d downto 0) \n\
);\n\
--end algo \n\
\n\
",
	  extraInCnt-1,inputCnt*bitSize-1,extraOutCnt-1,outputCnt*bitSize-1,
	  extraInCnt-1,inputCnt*bitSize-1,extraOutCnt-1,outputCnt*bitSize-1,
	  //	  inputCnt*bitSize-1, outputCnt*bitSize-1,
	  extraInCnt-1,inputCnt*bitSize-1, 
	  extraInCnt-1,inputCnt*bitSize-1, 
	  inputCnt*bitSize-1, outputCnt*bitSize-1
	  );
  
  fprintf(fp,"extraInGen : for i in 1 to %d generate\n\
begin\n\
IOBUF : IBUF port map (I=>EXTRA_INPUT_PADS(i), O=>EXTRA_INPUT_SIG3(i));\n\
process(clk)\n\
begin\n\
    if rising_edge(clk) then\n\
        EXTRA_INPUT_SIG2(i) <= EXTRA_INPUT_SIG3(i);\n\
        EXTRA_INPUT_SIG(i) <= EXTRA_INPUT_SIG2(i);\n\
    end if;\n\
 end process;\n\
end generate;\n\
\n\
",
	  extraInCnt-1);

  fprintf(fp,"inGen : for i in 0 to %d generate\n\
begin\n\
IOBUF : IBUF port map (I=>INPUT_PADS(i), O=>INPUT_SIG3(i));\n\
process(clk)\n\
begin\n\
    if rising_edge(clk) then\n\
        INPUT_SIG2(i) <= INPUT_SIG3(i);\n\
        INPUT_SIG(i) <= INPUT_SIG2(i);\n\
    end if;\n\
 end process;\n\
end generate;\n\
\n\
",
	  inputCnt*bitSize-1);
  fprintf(fp,"extraOutGen : for i in 0 to %d generate\n\
begin\n\
IOBUF : OBUF port map (I=>EXTRA_OUTPUT_SIG(i), O=>EXTRA_OUTPUT_PADS(i));\n\
end generate;\n\
\n\
",
	  extraOutCnt-1);
  fprintf(fp,"outGen : for i in 0 to %d generate\n\
begin\n\
IOBUF : OBUF port map (I=>OUTPUT_SIG(i), O=>OUTPUT_PADS(i));\n\
end generate;\n\
\n\
",
	  outputCnt*bitSize-1);
 

  fprintf(fp,"	\
\n\
IBUF_CLK : IBUFG    port map (I=>EXTRA_INPUT_PADS(0), O=>clk);\n\
\n\
end behavioral;\n\
");

  fclose(fp);



  cout << "done." << endl;
  return 0;  
}
